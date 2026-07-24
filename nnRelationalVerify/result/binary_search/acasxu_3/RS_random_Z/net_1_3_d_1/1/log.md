## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_3.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 9158.90444504786


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188)
1: (-5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188)
2: (-7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891)
3: (-2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812)
4: (-8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906)

## BASE Result
execution time: IAR + LP analysis = 1.13 + 2.11 = 3.24 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -9164.4112857, upper bound: 9164.4112857


# Binary Search by BASE starts (time budget: 1196.76 seconds, max iter: 100)

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
Binary search time: 67.50 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 1129.26 seconds

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.9573698, upper bound: 9163.9573770
time: 2.42 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.9573770, upper bound: 9163.9573698
time: 0.77 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 3.21 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 3.21
Output dim: 0, lower bound: -9163.9573698, upper bound: 9163.9573770
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 3.21
Output dim: 0, lower bound: -9163.9573770, upper bound: 9163.9573698

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.9128423, upper bound: 9163.9131379
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.9124800, upper bound: 9163.9131238
time: 0.83 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.9573669, upper bound: 9163.9573691
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.9573669, upper bound: 9163.9573691
time: 0.75 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.51 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.51
Output dim: 0, lower bound: -9163.9128423, upper bound: 9163.9131379
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.51
Output dim: 0, lower bound: -9163.9124800, upper bound: 9163.9131238
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.51
Output dim: 0, lower bound: -9163.9573669, upper bound: 9163.9573691
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.51
Output dim: 0, lower bound: -9163.9573669, upper bound: 9163.9573691

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.9127005, upper bound: 9163.9129274
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.9123090, upper bound: 9163.9129813
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.8139921, upper bound: 9163.8148293
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.8140062, upper bound: 9163.8148163
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.9322612, upper bound: 9163.9323051
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.9322612, upper bound: 9163.9323051
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.5332071, upper bound: 9163.5338597
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.5334062, upper bound: 9163.5332771
time: 0.79 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.89 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.89
Output dim: 0, lower bound: -9163.9127005, upper bound: 9163.9129274
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.89
Output dim: 0, lower bound: -9163.9123090, upper bound: 9163.9129813
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.89
Output dim: 0, lower bound: -9163.8139921, upper bound: 9163.8148293
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.89
Output dim: 0, lower bound: -9163.8140062, upper bound: 9163.8148163
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.89
Output dim: 0, lower bound: -9163.9322612, upper bound: 9163.9323051
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.89
Output dim: 0, lower bound: -9163.9322612, upper bound: 9163.9323051
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.89
Output dim: 0, lower bound: -9163.5332071, upper bound: 9163.5338597
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.89
Output dim: 0, lower bound: -9163.5334062, upper bound: 9163.5332771

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.9123243, upper bound: 9163.9129223
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.9127005, upper bound: 9163.9123904
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.8823734, upper bound: 9163.8831665
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.8823734, upper bound: 9163.8823734
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.4503729, upper bound: 9163.4528384
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.4503729, upper bound: 9163.4506413
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6257373, upper bound: 9161.6257373
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6257373, upper bound: 9161.6257373
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.0060403, upper bound: 9160.0067033
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.0060403, upper bound: 9160.0067033
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.2209178, upper bound: 9162.2209138
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.2209178, upper bound: 9162.2209138
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.2955100, upper bound: 9161.2952189
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.2955100, upper bound: 9161.2952189
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.0172311, upper bound: 9163.0176828
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.0177867, upper bound: 9163.0172311
time: 0.76 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.74 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -9163.9123243, upper bound: 9163.9129223
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -9163.9127005, upper bound: 9163.9123904
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -9163.8823734, upper bound: 9163.8831665
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -9163.8823734, upper bound: 9163.8823734
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -9163.4503729, upper bound: 9163.4528384
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -9163.4503729, upper bound: 9163.4506413
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -9161.6257373, upper bound: 9161.6257373
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -9161.6257373, upper bound: 9161.6257373
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -9160.0060403, upper bound: 9160.0067033
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -9160.0060403, upper bound: 9160.0067033
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -9162.2209178, upper bound: 9162.2209138
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -9162.2209178, upper bound: 9162.2209138
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -9161.2955100, upper bound: 9161.2952189
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -9161.2955100, upper bound: 9161.2952189
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -9163.0172311, upper bound: 9163.0176828
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -9163.0177867, upper bound: 9163.0172311

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7188704, upper bound: 9160.7190979
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7188681, upper bound: 9160.7190979
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.8453311, upper bound: 9163.8453206
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.8453311, upper bound: 9163.8453206
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.8787837, upper bound: 9163.8802048
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.8787837, upper bound: 9163.8807481
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.8605085, upper bound: 9162.8605085
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.8605085, upper bound: 9162.8605085
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.4503008, upper bound: 9163.4527519
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.4503008, upper bound: 9163.4516130
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.4503729, upper bound: 9163.4506413
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.4503729, upper bound: 9163.4503729
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6100044, upper bound: 9161.6100044
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6100044, upper bound: 9161.6100044
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3379914, upper bound: 9160.3379914
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3379914, upper bound: 9160.3379914
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9157.7582950, upper bound: 9157.7582950
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9157.7582950, upper bound: 9157.7589580
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.0060366, upper bound: 9160.0066989
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.0060366, upper bound: 9160.0060366
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.4313163, upper bound: 9161.4313163
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.4313163, upper bound: 9161.4313163
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.2209138, upper bound: 9162.2209138
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.2209178, upper bound: 9162.2209138
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1887430, upper bound: 9161.1884966
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1884966, upper bound: 9161.1884966
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.2886666, upper bound: 9161.2882143
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.2885975, upper bound: 9161.2882143
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.9301749, upper bound: 9162.9306309
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.9301749, upper bound: 9162.9301749
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.0177867, upper bound: 9163.0172311
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.0177802, upper bound: 9163.0172311
time: 0.89 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.09 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 0, lower bound: -9160.7188704, upper bound: 9160.7190979
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 0, lower bound: -9160.7188681, upper bound: 9160.7190979
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 0, lower bound: -9163.8453311, upper bound: 9163.8453206
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 0, lower bound: -9163.8453311, upper bound: 9163.8453206
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 0, lower bound: -9163.8787837, upper bound: 9163.8802048
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 0, lower bound: -9163.8787837, upper bound: 9163.8807481
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 0, lower bound: -9162.8605085, upper bound: 9162.8605085
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 0, lower bound: -9162.8605085, upper bound: 9162.8605085
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 0, lower bound: -9163.4503008, upper bound: 9163.4527519
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 0, lower bound: -9163.4503008, upper bound: 9163.4516130
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 0, lower bound: -9163.4503729, upper bound: 9163.4506413
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 0, lower bound: -9163.4503729, upper bound: 9163.4503729
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 0, lower bound: -9161.6100044, upper bound: 9161.6100044
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 0, lower bound: -9161.6100044, upper bound: 9161.6100044
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 0, lower bound: -9160.3379914, upper bound: 9160.3379914
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 0, lower bound: -9160.3379914, upper bound: 9160.3379914
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.09
Output dim: 0, lower bound: -9157.7582950, upper bound: 9157.7582950
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.09
Output dim: 0, lower bound: -9157.7582950, upper bound: 9157.7589580
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 0, lower bound: -9160.0060366, upper bound: 9160.0066989
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 0, lower bound: -9160.0060366, upper bound: 9160.0060366
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 0, lower bound: -9161.4313163, upper bound: 9161.4313163
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 0, lower bound: -9161.4313163, upper bound: 9161.4313163
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 0, lower bound: -9162.2209138, upper bound: 9162.2209138
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 0, lower bound: -9162.2209178, upper bound: 9162.2209138
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 0, lower bound: -9161.1887430, upper bound: 9161.1884966
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 0, lower bound: -9161.1884966, upper bound: 9161.1884966
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 0, lower bound: -9161.2886666, upper bound: 9161.2882143
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 0, lower bound: -9161.2885975, upper bound: 9161.2882143
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 0, lower bound: -9162.9301749, upper bound: 9162.9306309
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 0, lower bound: -9162.9301749, upper bound: 9162.9301749
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 0, lower bound: -9163.0177867, upper bound: 9163.0172311
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.09
Output dim: 0, lower bound: -9163.0177802, upper bound: 9163.0172311

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.5231016, upper bound: 9160.5237638
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.5231172, upper bound: 9160.5230228
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9157.3316193, upper bound: 9157.3316193
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9157.3316193, upper bound: 9157.3316193
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.7873676, upper bound: 9163.7873676
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.7873780, upper bound: 9163.7873676
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.8451795, upper bound: 9163.8451704
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.8451704, upper bound: 9163.8451704
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.8780606, upper bound: 9163.8793516
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.8780606, upper bound: 9163.8780606
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6273769, upper bound: 9161.6273769
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6273769, upper bound: 9161.6273769
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.7843371, upper bound: 9162.7843371
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.7843371, upper bound: 9162.7843371
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6721422, upper bound: 9161.6721422
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6721422, upper bound: 9161.6721422
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.7448617, upper bound: 9162.7454635
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.7448617, upper bound: 9162.7459039
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.4501484, upper bound: 9163.4503391
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.4501484, upper bound: 9163.4514946
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.4503729, upper bound: 9163.4503729
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.4503729, upper bound: 9163.4506413
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.0894383, upper bound: 9162.0894383
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.0894383, upper bound: 9162.0894383
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6100044, upper bound: 9161.6100044
time: 1.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6100044, upper bound: 9161.6100044
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6095745, upper bound: 9161.6095745
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6095745, upper bound: 9161.6095745
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3378620, upper bound: 9160.3378620
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3378620, upper bound: 9160.3378620
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3378620, upper bound: 9160.3378620
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3378620, upper bound: 9160.3378620
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.0008156, upper bound: 9160.0014597
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.0008156, upper bound: 9160.0014511
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9157.7582913, upper bound: 9157.7582913
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9157.7582913, upper bound: 9157.7582913
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.4313163, upper bound: 9161.4313163
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.4313163, upper bound: 9161.4313163
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.4313163, upper bound: 9161.4313163
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.4313163, upper bound: 9161.4313163
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.2204008, upper bound: 9162.2204008
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.2204008, upper bound: 9162.2204008
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.2204008, upper bound: 9162.2204008
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.2204190, upper bound: 9162.2204008
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9158.5671535, upper bound: 9158.5655758
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9158.5676921, upper bound: 9158.5652151
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1546346, upper bound: 9161.1546346
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1546346, upper bound: 9161.1546346
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.2882143, upper bound: 9161.2882143
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.2885877, upper bound: 9161.2882143
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.2885608, upper bound: 9161.2882143
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.2885975, upper bound: 9161.2882143
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.5951017, upper bound: 9162.5952351
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.5954310, upper bound: 9162.5953221
time: 1.22 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.5263263, upper bound: 9160.5263263
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.5263263, upper bound: 9160.5263263
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.7658320, upper bound: 9162.7650030
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.7650030, upper bound: 9162.7650030
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.0175926, upper bound: 9163.0170511
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.0172887, upper bound: 9163.0170511
time: 1.37 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.64 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9160.5231016, upper bound: 9160.5237638
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9160.5231172, upper bound: 9160.5230228
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.64
Output dim: 0, lower bound: -9157.3316193, upper bound: 9157.3316193
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.64
Output dim: 0, lower bound: -9157.3316193, upper bound: 9157.3316193
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9163.7873676, upper bound: 9163.7873676
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9163.7873780, upper bound: 9163.7873676
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9163.8451795, upper bound: 9163.8451704
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9163.8451704, upper bound: 9163.8451704
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9163.8780606, upper bound: 9163.8793516
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9163.8780606, upper bound: 9163.8780606
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9161.6273769, upper bound: 9161.6273769
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9161.6273769, upper bound: 9161.6273769
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9162.7843371, upper bound: 9162.7843371
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9162.7843371, upper bound: 9162.7843371
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9161.6721422, upper bound: 9161.6721422
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9161.6721422, upper bound: 9161.6721422
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9162.7448617, upper bound: 9162.7454635
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9162.7448617, upper bound: 9162.7459039
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9163.4501484, upper bound: 9163.4503391
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9163.4501484, upper bound: 9163.4514946
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9163.4503729, upper bound: 9163.4503729
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9163.4503729, upper bound: 9163.4506413
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9162.0894383, upper bound: 9162.0894383
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9162.0894383, upper bound: 9162.0894383
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9161.6100044, upper bound: 9161.6100044
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9161.6100044, upper bound: 9161.6100044
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9161.6095745, upper bound: 9161.6095745
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9161.6095745, upper bound: 9161.6095745
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9160.3378620, upper bound: 9160.3378620
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9160.3378620, upper bound: 9160.3378620
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9160.3378620, upper bound: 9160.3378620
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9160.3378620, upper bound: 9160.3378620
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9160.0008156, upper bound: 9160.0014597
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9160.0008156, upper bound: 9160.0014511
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.64
Output dim: 0, lower bound: -9157.7582913, upper bound: 9157.7582913
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.64
Output dim: 0, lower bound: -9157.7582913, upper bound: 9157.7582913
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9161.4313163, upper bound: 9161.4313163
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9161.4313163, upper bound: 9161.4313163
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9161.4313163, upper bound: 9161.4313163
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9161.4313163, upper bound: 9161.4313163
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9162.2204008, upper bound: 9162.2204008
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9162.2204008, upper bound: 9162.2204008
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9162.2204008, upper bound: 9162.2204008
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9162.2204190, upper bound: 9162.2204008
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.64
Output dim: 0, lower bound: -9158.5671535, upper bound: 9158.5655758
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.64
Output dim: 0, lower bound: -9158.5676921, upper bound: 9158.5652151
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9161.1546346, upper bound: 9161.1546346
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9161.1546346, upper bound: 9161.1546346
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9161.2882143, upper bound: 9161.2882143
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9161.2885877, upper bound: 9161.2882143
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9161.2885608, upper bound: 9161.2882143
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9161.2885975, upper bound: 9161.2882143
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9162.5951017, upper bound: 9162.5952351
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9162.5954310, upper bound: 9162.5953221
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9160.5263263, upper bound: 9160.5263263
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9160.5263263, upper bound: 9160.5263263
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9162.7658320, upper bound: 9162.7650030
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9162.7650030, upper bound: 9162.7650030
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9163.0175926, upper bound: 9163.0170511
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.64
Output dim: 0, lower bound: -9163.0172887, upper bound: 9163.0170511

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.2139422, upper bound: 9160.2149201
time: 2.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.2139357, upper bound: 9160.2149201
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.5231172, upper bound: 9160.5230228
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.5230228, upper bound: 9160.5230228
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.7812035, upper bound: 9163.7812035
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.7812035, upper bound: 9163.7812035
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.7812380, upper bound: 9163.7812035
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.7812384, upper bound: 9163.7812035
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.8451795, upper bound: 9163.8451704
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.8451704, upper bound: 9163.8451704
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.3131147, upper bound: 9163.3131171
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.3131147, upper bound: 9163.3131617
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.8780606, upper bound: 9163.8785464
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.8780606, upper bound: 9163.8793516
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.3154210, upper bound: 9163.3154210
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.3154210, upper bound: 9163.3154210
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1240137, upper bound: 9161.1240137
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1240137, upper bound: 9161.1240137
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.3462287, upper bound: 9161.3462287
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.3462287, upper bound: 9161.3462287
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5577291, upper bound: 9161.5577291
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5577291, upper bound: 9161.5577291
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.7843371, upper bound: 9162.7843371
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.7843371, upper bound: 9162.7843371
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6024518, upper bound: 9161.6024518
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6024518, upper bound: 9161.6024518
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6712719, upper bound: 9161.6712719
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6712719, upper bound: 9161.6712719
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5679256, upper bound: 9161.5679256
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5679256, upper bound: 9161.5679256
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.7414853, upper bound: 9162.7425791
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.7414853, upper bound: 9162.7434261
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.4501484, upper bound: 9163.4503391
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.4501484, upper bound: 9163.4501484
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.4500524, upper bound: 9163.4500524
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.4500524, upper bound: 9163.4514089
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1978745, upper bound: 9161.1978745
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1978745, upper bound: 9161.1978745
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.0894383, upper bound: 9162.0895049
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.0894383, upper bound: 9162.0895049
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.0894383, upper bound: 9162.0894383
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.0894383, upper bound: 9162.0894383
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.0833025, upper bound: 9162.0833025
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.0833025, upper bound: 9162.0833025
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6100044, upper bound: 9161.6100044
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6100044, upper bound: 9161.6100044
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6099499, upper bound: 9161.6099499
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6099499, upper bound: 9161.6099499
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.4376666, upper bound: 9161.4376666
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.4376666, upper bound: 9161.4376666
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6095745, upper bound: 9161.6095745
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6095745, upper bound: 9161.6095745
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.0855109, upper bound: 9160.0855109
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.0855109, upper bound: 9160.0855109
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3378525, upper bound: 9160.3378525
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3378525, upper bound: 9160.3378525
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3378620, upper bound: 9160.3378620
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3378620, upper bound: 9160.3378620
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3378620, upper bound: 9160.3378620
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3378620, upper bound: 9160.3378620
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.0006733, upper bound: 9160.0006733
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.0006733, upper bound: 9160.0013696
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.0008156, upper bound: 9160.0014511
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.0008156, upper bound: 9160.0011573
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.4307447, upper bound: 9161.4307447
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.4307447, upper bound: 9161.4307447
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.2663475, upper bound: 9161.2663475
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.2663475, upper bound: 9161.2663475
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.4313163, upper bound: 9161.4313163
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.4313163, upper bound: 9161.4313163
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.2980107, upper bound: 9161.2980107
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.2980107, upper bound: 9161.2980107
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.2204008, upper bound: 9162.2204008
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.2204008, upper bound: 9162.2204008
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.4307447, upper bound: 9161.4307447
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.4307447, upper bound: 9161.4307447
time: 1.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.4307447, upper bound: 9161.4307447
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.4307447, upper bound: 9161.4307447
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.9321818, upper bound: 9161.9322129
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.9321664, upper bound: 9161.9321664
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1546346, upper bound: 9161.1546346
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1546346, upper bound: 9161.1546346
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1546346, upper bound: 9161.1546346
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1546346, upper bound: 9161.1546346
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.2881788, upper bound: 9161.2881788
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.2881788, upper bound: 9161.2881788
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6762488, upper bound: 9160.6757883
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6762488, upper bound: 9160.6757883
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.0517852, upper bound: 9161.0517852
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.0522187, upper bound: 9161.0517852
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.2870193, upper bound: 9161.2870193
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.2871459, upper bound: 9161.2870193
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.5951017, upper bound: 9162.5951017
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.5951017, upper bound: 9162.5952351
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.5952620, upper bound: 9162.5949204
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.5951205, upper bound: 9162.5951519
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.4761466, upper bound: 9160.4761466
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.4761466, upper bound: 9160.4761466
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.5029674, upper bound: 9160.5029674
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.5029674, upper bound: 9160.5029674
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.7463022, upper bound: 9162.7454631
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.7454631, upper bound: 9162.7454631
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.7328312, upper bound: 9162.7328312
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.7328312, upper bound: 9162.7328312
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.2867658, upper bound: 9159.2867631
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.2867658, upper bound: 9159.2867631
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.9404673, upper bound: 9161.9404673
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.9404673, upper bound: 9161.9404673
time: 0.69 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.84 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9160.2139422, upper bound: 9160.2149201
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9160.2139357, upper bound: 9160.2149201
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9160.5231172, upper bound: 9160.5230228
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9160.5230228, upper bound: 9160.5230228
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9163.7812035, upper bound: 9163.7812035
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9163.7812035, upper bound: 9163.7812035
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9163.7812380, upper bound: 9163.7812035
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9163.7812384, upper bound: 9163.7812035
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9163.8451795, upper bound: 9163.8451704
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9163.8451704, upper bound: 9163.8451704
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9163.3131147, upper bound: 9163.3131171
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9163.3131147, upper bound: 9163.3131617
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9163.8780606, upper bound: 9163.8785464
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9163.8780606, upper bound: 9163.8793516
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9163.3154210, upper bound: 9163.3154210
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9163.3154210, upper bound: 9163.3154210
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9161.1240137, upper bound: 9161.1240137
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9161.1240137, upper bound: 9161.1240137
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9161.3462287, upper bound: 9161.3462287
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9161.3462287, upper bound: 9161.3462287
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9161.5577291, upper bound: 9161.5577291
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9161.5577291, upper bound: 9161.5577291
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9162.7843371, upper bound: 9162.7843371
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9162.7843371, upper bound: 9162.7843371
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9161.6024518, upper bound: 9161.6024518
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9161.6024518, upper bound: 9161.6024518
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9161.6712719, upper bound: 9161.6712719
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9161.6712719, upper bound: 9161.6712719
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9161.5679256, upper bound: 9161.5679256
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9161.5679256, upper bound: 9161.5679256
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9162.7414853, upper bound: 9162.7425791
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9162.7414853, upper bound: 9162.7434261
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9163.4501484, upper bound: 9163.4503391
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9163.4501484, upper bound: 9163.4501484
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9163.4500524, upper bound: 9163.4500524
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9163.4500524, upper bound: 9163.4514089
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9161.1978745, upper bound: 9161.1978745
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9161.1978745, upper bound: 9161.1978745
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9162.0894383, upper bound: 9162.0895049
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9162.0894383, upper bound: 9162.0895049
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9162.0894383, upper bound: 9162.0894383
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9162.0894383, upper bound: 9162.0894383
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9162.0833025, upper bound: 9162.0833025
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9162.0833025, upper bound: 9162.0833025
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9161.6100044, upper bound: 9161.6100044
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9161.6100044, upper bound: 9161.6100044
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9161.6099499, upper bound: 9161.6099499
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9161.6099499, upper bound: 9161.6099499
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9161.4376666, upper bound: 9161.4376666
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9161.4376666, upper bound: 9161.4376666
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9161.6095745, upper bound: 9161.6095745
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9161.6095745, upper bound: 9161.6095745
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9160.0855109, upper bound: 9160.0855109
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9160.0855109, upper bound: 9160.0855109
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9160.3378525, upper bound: 9160.3378525
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9160.3378525, upper bound: 9160.3378525
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9160.3378620, upper bound: 9160.3378620
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9160.3378620, upper bound: 9160.3378620
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9160.3378620, upper bound: 9160.3378620
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9160.3378620, upper bound: 9160.3378620
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9160.0006733, upper bound: 9160.0006733
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9160.0006733, upper bound: 9160.0013696
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9160.0008156, upper bound: 9160.0014511
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9160.0008156, upper bound: 9160.0011573
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9161.4307447, upper bound: 9161.4307447
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9161.4307447, upper bound: 9161.4307447
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9161.2663475, upper bound: 9161.2663475
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9161.2663475, upper bound: 9161.2663475
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9161.4313163, upper bound: 9161.4313163
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9161.4313163, upper bound: 9161.4313163
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9161.2980107, upper bound: 9161.2980107
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9161.2980107, upper bound: 9161.2980107
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9162.2204008, upper bound: 9162.2204008
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9162.2204008, upper bound: 9162.2204008
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9161.4307447, upper bound: 9161.4307447
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9161.4307447, upper bound: 9161.4307447
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9161.4307447, upper bound: 9161.4307447
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9161.4307447, upper bound: 9161.4307447
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9161.9321818, upper bound: 9161.9322129
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9161.9321664, upper bound: 9161.9321664
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9161.1546346, upper bound: 9161.1546346
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9161.1546346, upper bound: 9161.1546346
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9161.1546346, upper bound: 9161.1546346
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9161.1546346, upper bound: 9161.1546346
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9161.2881788, upper bound: 9161.2881788
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9161.2881788, upper bound: 9161.2881788
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9160.6762488, upper bound: 9160.6757883
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9160.6762488, upper bound: 9160.6757883
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9161.0517852, upper bound: 9161.0517852
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9161.0522187, upper bound: 9161.0517852
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9161.2870193, upper bound: 9161.2870193
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9161.2871459, upper bound: 9161.2870193
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9162.5951017, upper bound: 9162.5951017
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9162.5951017, upper bound: 9162.5952351
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9162.5952620, upper bound: 9162.5949204
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9162.5951205, upper bound: 9162.5951519
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9160.4761466, upper bound: 9160.4761466
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9160.4761466, upper bound: 9160.4761466
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9160.5029674, upper bound: 9160.5029674
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9160.5029674, upper bound: 9160.5029674
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9162.7463022, upper bound: 9162.7454631
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9162.7454631, upper bound: 9162.7454631
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9162.7328312, upper bound: 9162.7328312
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9162.7328312, upper bound: 9162.7328312
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9159.2867658, upper bound: 9159.2867631
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9159.2867658, upper bound: 9159.2867631
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9161.9404673, upper bound: 9161.9404673
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -9161.9404673, upper bound: 9161.9404673

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.2128891, upper bound: 9160.2128347
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.2128777, upper bound: 9160.2138044
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.1923972, upper bound: 9160.1935270
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.1925201, upper bound: 9160.1923435
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3569752, upper bound: 9160.3569752
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3570940, upper bound: 9160.3569983
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.5230228, upper bound: 9160.5230228
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.5230228, upper bound: 9160.5230228
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.8250335, upper bound: 9161.8250335
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.8250335, upper bound: 9161.8250335
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.7812035, upper bound: 9163.7812035
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.7812035, upper bound: 9163.7812035
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.7812217, upper bound: 9163.7812035
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.7812380, upper bound: 9163.7812035
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.5297261, upper bound: 9163.5297261
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.5297783, upper bound: 9163.5297261
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.7872174, upper bound: 9163.7872174
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.7872265, upper bound: 9163.7872174
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.9632848, upper bound: 9161.9632848
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.9632848, upper bound: 9161.9632848
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6757713, upper bound: 9160.6757713
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6757713, upper bound: 9160.6757713
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.1554818, upper bound: 9163.1554818
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.1554818, upper bound: 9163.1554818
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.8565846, upper bound: 9162.8567032
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.8565846, upper bound: 9162.8567032
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.8093343, upper bound: 9163.8106022
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.8093343, upper bound: 9163.8103127
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.3154210, upper bound: 9163.3154210
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.3154210, upper bound: 9163.3154210
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.2956627, upper bound: 9162.2956627
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.2956627, upper bound: 9162.2956627
time: 1.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.0999054, upper bound: 9161.0999054
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.0999054, upper bound: 9161.0999054
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7894468, upper bound: 9160.7894468
time: 6.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7894468, upper bound: 9160.7894468
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.3462287, upper bound: 9161.3462287
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.3462287, upper bound: 9161.3462287
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7916323, upper bound: 9160.7916323
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7916323, upper bound: 9160.7916323
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5561330, upper bound: 9161.5561330
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5561330, upper bound: 9161.5561330
time: 0.79 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.93 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 0, lower bound: -9160.2128891, upper bound: 9160.2128347
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 0, lower bound: -9160.2128777, upper bound: 9160.2138044
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 0, lower bound: -9160.1923972, upper bound: 9160.1935270
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 0, lower bound: -9160.1925201, upper bound: 9160.1923435
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 0, lower bound: -9160.3569752, upper bound: 9160.3569752
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 0, lower bound: -9160.3570940, upper bound: 9160.3569983
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 0, lower bound: -9160.5230228, upper bound: 9160.5230228
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 0, lower bound: -9160.5230228, upper bound: 9160.5230228
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 0, lower bound: -9161.8250335, upper bound: 9161.8250335
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 0, lower bound: -9161.8250335, upper bound: 9161.8250335
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 0, lower bound: -9163.7812035, upper bound: 9163.7812035
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 0, lower bound: -9163.7812035, upper bound: 9163.7812035
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 0, lower bound: -9163.7812217, upper bound: 9163.7812035
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 0, lower bound: -9163.7812380, upper bound: 9163.7812035
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 0, lower bound: -9163.5297261, upper bound: 9163.5297261
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 0, lower bound: -9163.5297783, upper bound: 9163.5297261
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 0, lower bound: -9163.7872174, upper bound: 9163.7872174
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 0, lower bound: -9163.7872265, upper bound: 9163.7872174
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 0, lower bound: -9161.9632848, upper bound: 9161.9632848
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 0, lower bound: -9161.9632848, upper bound: 9161.9632848
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 0, lower bound: -9160.6757713, upper bound: 9160.6757713
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 0, lower bound: -9160.6757713, upper bound: 9160.6757713
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 0, lower bound: -9163.1554818, upper bound: 9163.1554818
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 0, lower bound: -9163.1554818, upper bound: 9163.1554818
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 0, lower bound: -9162.8565846, upper bound: 9162.8567032
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 0, lower bound: -9162.8565846, upper bound: 9162.8567032
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 0, lower bound: -9163.8093343, upper bound: 9163.8106022
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 0, lower bound: -9163.8093343, upper bound: 9163.8103127
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 0, lower bound: -9163.3154210, upper bound: 9163.3154210
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 0, lower bound: -9163.3154210, upper bound: 9163.3154210
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 0, lower bound: -9162.2956627, upper bound: 9162.2956627
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 0, lower bound: -9162.2956627, upper bound: 9162.2956627
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 0, lower bound: -9161.0999054, upper bound: 9161.0999054
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 0, lower bound: -9161.0999054, upper bound: 9161.0999054
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 0, lower bound: -9160.7894468, upper bound: 9160.7894468
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 0, lower bound: -9160.7894468, upper bound: 9160.7894468
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 0, lower bound: -9161.3462287, upper bound: 9161.3462287
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 0, lower bound: -9161.3462287, upper bound: 9161.3462287
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 0, lower bound: -9160.7916323, upper bound: 9160.7916323
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 0, lower bound: -9160.7916323, upper bound: 9160.7916323
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 0, lower bound: -9161.5561330, upper bound: 9161.5561330
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.93
Output dim: 0, lower bound: -9161.5561330, upper bound: 9161.5561330
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9161.5577291, upper bound: 9161.5577291
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9162.7843371, upper bound: 9162.7843371
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9162.7843371, upper bound: 9162.7843371
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9161.6024518, upper bound: 9161.6024518
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9161.6024518, upper bound: 9161.6024518
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9161.6712719, upper bound: 9161.6712719
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9161.6712719, upper bound: 9161.6712719
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9161.5679256, upper bound: 9161.5679256
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9161.5679256, upper bound: 9161.5679256
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9162.7414853, upper bound: 9162.7425791
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9162.7414853, upper bound: 9162.7434261
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9163.4501484, upper bound: 9163.4503391
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9163.4501484, upper bound: 9163.4501484
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9163.4500524, upper bound: 9163.4500524
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9163.4500524, upper bound: 9163.4514089
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9161.1978745, upper bound: 9161.1978745
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9161.1978745, upper bound: 9161.1978745
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9162.0894383, upper bound: 9162.0895049
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9162.0894383, upper bound: 9162.0895049
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9162.0894383, upper bound: 9162.0894383
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9162.0894383, upper bound: 9162.0894383
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9162.0833025, upper bound: 9162.0833025
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9162.0833025, upper bound: 9162.0833025
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9161.6100044, upper bound: 9161.6100044
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9161.6100044, upper bound: 9161.6100044
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9161.6099499, upper bound: 9161.6099499
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9161.6099499, upper bound: 9161.6099499
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9161.4376666, upper bound: 9161.4376666
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9161.4376666, upper bound: 9161.4376666
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9161.6095745, upper bound: 9161.6095745
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9161.6095745, upper bound: 9161.6095745
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9160.0855109, upper bound: 9160.0855109
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9160.0855109, upper bound: 9160.0855109
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9160.3378525, upper bound: 9160.3378525
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9160.3378525, upper bound: 9160.3378525
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9160.3378620, upper bound: 9160.3378620
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9160.3378620, upper bound: 9160.3378620
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9160.3378620, upper bound: 9160.3378620
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9160.3378620, upper bound: 9160.3378620
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9160.0006733, upper bound: 9160.0006733
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9160.0006733, upper bound: 9160.0013696
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9160.0008156, upper bound: 9160.0014511
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9160.0008156, upper bound: 9160.0011573
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9161.4307447, upper bound: 9161.4307447
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9161.4307447, upper bound: 9161.4307447
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9161.2663475, upper bound: 9161.2663475
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9161.2663475, upper bound: 9161.2663475
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9161.4313163, upper bound: 9161.4313163
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9161.4313163, upper bound: 9161.4313163
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9161.2980107, upper bound: 9161.2980107
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9161.2980107, upper bound: 9161.2980107
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9162.2204008, upper bound: 9162.2204008
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9162.2204008, upper bound: 9162.2204008
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9161.4307447, upper bound: 9161.4307447
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9161.4307447, upper bound: 9161.4307447
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9161.4307447, upper bound: 9161.4307447
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9161.4307447, upper bound: 9161.4307447
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9161.9321818, upper bound: 9161.9322129
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9161.9321664, upper bound: 9161.9321664
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9161.1546346, upper bound: 9161.1546346
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9161.1546346, upper bound: 9161.1546346
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9161.1546346, upper bound: 9161.1546346
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9161.1546346, upper bound: 9161.1546346
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9161.2881788, upper bound: 9161.2881788
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9161.2881788, upper bound: 9161.2881788
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9160.6762488, upper bound: 9160.6757883
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9160.6762488, upper bound: 9160.6757883
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9161.0517852, upper bound: 9161.0517852
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9161.0522187, upper bound: 9161.0517852
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9161.2870193, upper bound: 9161.2870193
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9161.2871459, upper bound: 9161.2870193
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9162.5951017, upper bound: 9162.5951017
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9162.5951017, upper bound: 9162.5952351
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9162.5952620, upper bound: 9162.5949204
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9162.5951205, upper bound: 9162.5951519
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9160.4761466, upper bound: 9160.4761466
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9160.4761466, upper bound: 9160.4761466
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9160.5029674, upper bound: 9160.5029674
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9160.5029674, upper bound: 9160.5029674
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9162.7463022, upper bound: 9162.7454631
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9162.7454631, upper bound: 9162.7454631
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9162.7328312, upper bound: 9162.7328312
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9162.7328312, upper bound: 9162.7328312
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9159.2867658, upper bound: 9159.2867631
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9159.2867658, upper bound: 9159.2867631
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9161.9404673, upper bound: 9161.9404673
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.93
Output dim: 0, lower bound: -9161.9404673, upper bound: 9161.9404673
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=11463.63671875
rel_dist={0: [-9164.411285672151, 9164.411285672155]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9164.3168700, upper bound: 9164.3166758
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9164.3166758, upper bound: 9164.3168700
time: 0.75 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.61 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.61
Output dim: 0, lower bound: -9164.3168700, upper bound: 9164.3166758
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.61
Output dim: 0, lower bound: -9164.3166758, upper bound: 9164.3168700

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9164.2569413, upper bound: 9164.2554626
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9164.2567336, upper bound: 9164.2564768
time: 0.83 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6600009, upper bound: 9161.6602949
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6600009, upper bound: 9161.6602949
time: 0.76 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.58 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.58
Output dim: 0, lower bound: -9164.2569413, upper bound: 9164.2554626
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.58
Output dim: 0, lower bound: -9164.2567336, upper bound: 9164.2564768
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.58
Output dim: 0, lower bound: -9161.6600009, upper bound: 9161.6602949
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.58
Output dim: 0, lower bound: -9161.6600009, upper bound: 9161.6602949

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.8244118, upper bound: 9163.8244889
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.8276718, upper bound: 9163.8240561
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9164.2566798, upper bound: 9164.2561368
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9164.2551626, upper bound: 9164.2564384
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.2880895, upper bound: 9161.2893179
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.2882488, upper bound: 9161.2880886
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.0989338, upper bound: 9161.0989269
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.0989319, upper bound: 9161.0989846
time: 0.90 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.98 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.98
Output dim: 0, lower bound: -9163.8244118, upper bound: 9163.8244889
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.98
Output dim: 0, lower bound: -9163.8276718, upper bound: 9163.8240561
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.98
Output dim: 0, lower bound: -9164.2566798, upper bound: 9164.2561368
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.98
Output dim: 0, lower bound: -9164.2551626, upper bound: 9164.2564384
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.98
Output dim: 0, lower bound: -9161.2880895, upper bound: 9161.2893179
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.98
Output dim: 0, lower bound: -9161.2882488, upper bound: 9161.2880886
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.98
Output dim: 0, lower bound: -9161.0989338, upper bound: 9161.0989269
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.98
Output dim: 0, lower bound: -9161.0989319, upper bound: 9161.0989846

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6225883, upper bound: 9162.6220515
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6225528, upper bound: 9162.6220515
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.7774791, upper bound: 9163.7729294
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.7740426, upper bound: 9163.7729294
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.6808024, upper bound: 9163.6810595
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.6812339, upper bound: 9163.6804337
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9164.2551626, upper bound: 9164.2564384
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9164.2551626, upper bound: 9164.2563808
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.0803642, upper bound: 9161.0810558
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.0803642, upper bound: 9161.0810558
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.0621632, upper bound: 9161.0621632
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.0621632, upper bound: 9161.0621632
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6346803, upper bound: 9160.6349535
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6348997, upper bound: 9160.6347154
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.8634278, upper bound: 9160.8632669
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.8634278, upper bound: 9160.8632763
time: 0.82 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.78 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -9162.6225883, upper bound: 9162.6220515
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -9162.6225528, upper bound: 9162.6220515
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -9163.7774791, upper bound: 9163.7729294
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -9163.7740426, upper bound: 9163.7729294
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -9163.6808024, upper bound: 9163.6810595
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -9163.6812339, upper bound: 9163.6804337
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -9164.2551626, upper bound: 9164.2564384
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -9164.2551626, upper bound: 9164.2563808
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -9161.0803642, upper bound: 9161.0810558
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -9161.0803642, upper bound: 9161.0810558
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -9161.0621632, upper bound: 9161.0621632
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -9161.0621632, upper bound: 9161.0621632
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -9160.6346803, upper bound: 9160.6349535
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -9160.6348997, upper bound: 9160.6347154
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -9160.8634278, upper bound: 9160.8632669
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -9160.8634278, upper bound: 9160.8632763

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6220515, upper bound: 9162.6220515
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6225883, upper bound: 9162.6220515
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.2025817, upper bound: 9162.2025817
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.2025817, upper bound: 9162.2025817
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.7774791, upper bound: 9163.7729294
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.7768105, upper bound: 9163.7729294
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.7740254, upper bound: 9163.7729104
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.7729104, upper bound: 9163.7729104
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.9553413, upper bound: 9161.9550329
time: 2.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.9553137, upper bound: 9161.9550329
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.6656988, upper bound: 9163.6656988
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.6656988, upper bound: 9163.6659218
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9164.0934086, upper bound: 9164.0939411
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9164.0934086, upper bound: 9164.0942278
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9164.1805002, upper bound: 9164.1809253
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9164.1805976, upper bound: 9164.1813299
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.0450496, upper bound: 9161.0456755
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.0450570, upper bound: 9161.0450496
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.0450496, upper bound: 9161.0456755
time: 2.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.0450570, upper bound: 9161.0450496
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.0277659, upper bound: 9161.0277659
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.0277659, upper bound: 9161.0277659
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3759134, upper bound: 9160.3755056
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3758393, upper bound: 9160.3755056
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9158.5384662, upper bound: 9158.5384662
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9158.5384662, upper bound: 9158.5384662
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6343673, upper bound: 9160.6342184
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6342184, upper bound: 9160.6342184
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.8633891, upper bound: 9160.8632669
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.8634278, upper bound: 9160.8632669
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.8323261, upper bound: 9160.8323938
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.8323414, upper bound: 9160.8323669
time: 0.69 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.06 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -9162.6220515, upper bound: 9162.6220515
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -9162.6225883, upper bound: 9162.6220515
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -9162.2025817, upper bound: 9162.2025817
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -9162.2025817, upper bound: 9162.2025817
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -9163.7774791, upper bound: 9163.7729294
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -9163.7768105, upper bound: 9163.7729294
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -9163.7740254, upper bound: 9163.7729104
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -9163.7729104, upper bound: 9163.7729104
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -9161.9553413, upper bound: 9161.9550329
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -9161.9553137, upper bound: 9161.9550329
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -9163.6656988, upper bound: 9163.6656988
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -9163.6656988, upper bound: 9163.6659218
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -9164.0934086, upper bound: 9164.0939411
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -9164.0934086, upper bound: 9164.0942278
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -9164.1805002, upper bound: 9164.1809253
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -9164.1805976, upper bound: 9164.1813299
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -9161.0450496, upper bound: 9161.0456755
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -9161.0450570, upper bound: 9161.0450496
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -9161.0450496, upper bound: 9161.0456755
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -9161.0450570, upper bound: 9161.0450496
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -9161.0277659, upper bound: 9161.0277659
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -9161.0277659, upper bound: 9161.0277659
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -9160.3759134, upper bound: 9160.3755056
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -9160.3758393, upper bound: 9160.3755056
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.06
Output dim: 0, lower bound: -9158.5384662, upper bound: 9158.5384662
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.06
Output dim: 0, lower bound: -9158.5384662, upper bound: 9158.5384662
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -9160.6343673, upper bound: 9160.6342184
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -9160.6342184, upper bound: 9160.6342184
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -9160.8633891, upper bound: 9160.8632669
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -9160.8634278, upper bound: 9160.8632669
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -9160.8323261, upper bound: 9160.8323938
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 0, lower bound: -9160.8323414, upper bound: 9160.8323669

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6080230, upper bound: 9161.6080230
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6080230, upper bound: 9161.6080230
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6088210, upper bound: 9162.6082637
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6082637, upper bound: 9162.6082637
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.1996168, upper bound: 9162.1996168
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.1996168, upper bound: 9162.1996168
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.0795356, upper bound: 9162.0795356
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.0795356, upper bound: 9162.0795356
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.8258853, upper bound: 9161.8258853
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.8258853, upper bound: 9161.8258853
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.7752636, upper bound: 9163.7729294
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.7768105, upper bound: 9163.7729294
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.7729104, upper bound: 9163.7729104
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.7740254, upper bound: 9163.7729104
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.7269984, upper bound: 9163.7269984
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.7269984, upper bound: 9163.7269984
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.9553413, upper bound: 9161.9550329
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.9553207, upper bound: 9161.9550329
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6929558, upper bound: 9161.6927784
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6929558, upper bound: 9161.6927784
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.6656988, upper bound: 9163.6656988
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.6656988, upper bound: 9163.6656988
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.2258756, upper bound: 9163.2258756
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.2270806, upper bound: 9163.2266346
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9164.0465074, upper bound: 9164.0468927
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9164.0465074, upper bound: 9164.0470523
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9164.0934086, upper bound: 9164.0942153
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9164.0934086, upper bound: 9164.0942278
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9164.1394838, upper bound: 9164.1397593
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9164.1394838, upper bound: 9164.1397593
time: 1.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9164.1798082, upper bound: 9164.1797214
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9164.1798177, upper bound: 9164.1805600
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9158.6584232, upper bound: 9158.6647442
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9158.6584232, upper bound: 9158.6642130
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.0450568, upper bound: 9161.0450496
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.0450570, upper bound: 9161.0450496
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.0450496, upper bound: 9161.0456755
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.0450496, upper bound: 9161.0450496
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.0450570, upper bound: 9161.0450496
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.0450496, upper bound: 9161.0450496
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.0277659, upper bound: 9161.0277659
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.0277659, upper bound: 9161.0277659
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.8027331, upper bound: 9160.8027331
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.8027331, upper bound: 9160.8027331
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3744307, upper bound: 9160.3740274
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3740274, upper bound: 9160.3740274
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3499705, upper bound: 9160.3499705
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3502980, upper bound: 9160.3499705
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6033496, upper bound: 9160.6033496
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6034969, upper bound: 9160.6033496
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6341477, upper bound: 9160.6341477
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6341477, upper bound: 9160.6341547
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.8633840, upper bound: 9160.8632669
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.8633891, upper bound: 9160.8632669
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3445649, upper bound: 9160.3446427
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3445649, upper bound: 9160.3445649
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.2864739, upper bound: 9160.2867220
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.2864739, upper bound: 9160.2864739
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.8322781, upper bound: 9160.8323669
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.8323414, upper bound: 9160.8322185
time: 0.76 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.15 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9161.6080230, upper bound: 9161.6080230
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9161.6080230, upper bound: 9161.6080230
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9162.6088210, upper bound: 9162.6082637
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9162.6082637, upper bound: 9162.6082637
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9162.1996168, upper bound: 9162.1996168
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9162.1996168, upper bound: 9162.1996168
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9162.0795356, upper bound: 9162.0795356
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9162.0795356, upper bound: 9162.0795356
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9161.8258853, upper bound: 9161.8258853
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9161.8258853, upper bound: 9161.8258853
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9163.7752636, upper bound: 9163.7729294
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9163.7768105, upper bound: 9163.7729294
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9163.7729104, upper bound: 9163.7729104
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9163.7740254, upper bound: 9163.7729104
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9163.7269984, upper bound: 9163.7269984
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9163.7269984, upper bound: 9163.7269984
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9161.9553413, upper bound: 9161.9550329
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9161.9553207, upper bound: 9161.9550329
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9161.6929558, upper bound: 9161.6927784
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9161.6929558, upper bound: 9161.6927784
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9163.6656988, upper bound: 9163.6656988
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9163.6656988, upper bound: 9163.6656988
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9163.2258756, upper bound: 9163.2258756
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9163.2270806, upper bound: 9163.2266346
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9164.0465074, upper bound: 9164.0468927
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9164.0465074, upper bound: 9164.0470523
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9164.0934086, upper bound: 9164.0942153
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9164.0934086, upper bound: 9164.0942278
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9164.1394838, upper bound: 9164.1397593
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9164.1394838, upper bound: 9164.1397593
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9164.1798082, upper bound: 9164.1797214
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9164.1798177, upper bound: 9164.1805600
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.15
Output dim: 0, lower bound: -9158.6584232, upper bound: 9158.6647442
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.15
Output dim: 0, lower bound: -9158.6584232, upper bound: 9158.6642130
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9161.0450568, upper bound: 9161.0450496
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9161.0450570, upper bound: 9161.0450496
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9161.0450496, upper bound: 9161.0456755
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9161.0450496, upper bound: 9161.0450496
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9161.0450570, upper bound: 9161.0450496
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9161.0450496, upper bound: 9161.0450496
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9161.0277659, upper bound: 9161.0277659
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9161.0277659, upper bound: 9161.0277659
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9160.8027331, upper bound: 9160.8027331
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9160.8027331, upper bound: 9160.8027331
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9160.3744307, upper bound: 9160.3740274
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9160.3740274, upper bound: 9160.3740274
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9160.3499705, upper bound: 9160.3499705
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9160.3502980, upper bound: 9160.3499705
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9160.6033496, upper bound: 9160.6033496
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9160.6034969, upper bound: 9160.6033496
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9160.6341477, upper bound: 9160.6341477
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9160.6341477, upper bound: 9160.6341547
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9160.8633840, upper bound: 9160.8632669
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9160.8633891, upper bound: 9160.8632669
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9160.3445649, upper bound: 9160.3446427
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9160.3445649, upper bound: 9160.3445649
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9160.2864739, upper bound: 9160.2867220
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9160.2864739, upper bound: 9160.2864739
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9160.8322781, upper bound: 9160.8323669
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.15
Output dim: 0, lower bound: -9160.8323414, upper bound: 9160.8322185

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6074284, upper bound: 9161.6074284
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6074284, upper bound: 9161.6074284
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6080230, upper bound: 9161.6080230
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6080230, upper bound: 9161.6080230
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6082637, upper bound: 9162.6082637
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6088210, upper bound: 9162.6082637
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.5027213, upper bound: 9162.5027213
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.5027213, upper bound: 9162.5027213
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.1977054, upper bound: 9162.1977054
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.1977054, upper bound: 9162.1977054
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.1996168, upper bound: 9162.1996168
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.1996168, upper bound: 9162.1996168
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.0792945, upper bound: 9162.0792945
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.0792945, upper bound: 9162.0792945
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.0791012, upper bound: 9162.0791012
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.0791012, upper bound: 9162.0791012
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6816563, upper bound: 9161.6816563
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6816563, upper bound: 9161.6816563
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.2449437, upper bound: 9161.2449437
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.2449437, upper bound: 9161.2449437
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.1730566, upper bound: 9163.1712227
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.1721719, upper bound: 9163.1712227
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.7767964, upper bound: 9163.7729104
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.7729104, upper bound: 9163.7729104
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.7269984, upper bound: 9163.7269984
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.7269984, upper bound: 9163.7269984
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3722621, upper bound: 9160.3722621
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3722621, upper bound: 9160.3722621
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.8258553, upper bound: 9161.8258553
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.8258553, upper bound: 9161.8258553
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6063753, upper bound: 9162.6063753
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6063753, upper bound: 9162.6063753
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.9123225, upper bound: 9161.9123225
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.9127277, upper bound: 9161.9123225
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.9552805, upper bound: 9161.9550259
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.9551477, upper bound: 9161.9550259
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.4465259, upper bound: 9161.4465259
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.4468778, upper bound: 9161.4465259
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5878441, upper bound: 9161.5878441
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5882220, upper bound: 9161.5878441
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.0798279, upper bound: 9163.0798279
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.0798279, upper bound: 9163.0798279
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.2961120, upper bound: 9162.2961120
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.2961120, upper bound: 9162.2961120
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.5589479, upper bound: 9160.5589479
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.5589479, upper bound: 9160.5589479
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.4563839, upper bound: 9162.4554320
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.4563839, upper bound: 9162.4557531
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.8230141, upper bound: 9163.8230141
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.8230141, upper bound: 9163.8230141
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9164.0441085, upper bound: 9164.0445693
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9164.0441085, upper bound: 9164.0457270
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9164.0925886, upper bound: 9164.0928381
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9164.0925886, upper bound: 9164.0934543
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.5424631, upper bound: 9162.5428021
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.5424631, upper bound: 9162.5429301
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.7199884, upper bound: 9163.7199884
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.7199884, upper bound: 9163.7205328
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9164.1393966, upper bound: 9164.1393966
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9164.1393966, upper bound: 9164.1397404
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9164.1387240, upper bound: 9164.1387240
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9164.1387240, upper bound: 9164.1387240
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.6182719, upper bound: 9163.6191574
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.6182719, upper bound: 9163.6192019
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.0450536, upper bound: 9161.0450496
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.0450568, upper bound: 9161.0450496
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.0391126, upper bound: 9161.0391126
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.0391126, upper bound: 9161.0391126
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.0441875, upper bound: 9161.0447157
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.0441875, upper bound: 9161.0441875
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7044105, upper bound: 9160.7044105
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.7044105, upper bound: 9160.7044105
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.0391164, upper bound: 9161.0391126
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.0391126, upper bound: 9161.0391126
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.0450496, upper bound: 9161.0450496
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.0450496, upper bound: 9161.0450496
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.0267527, upper bound: 9161.0267527
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.0267527, upper bound: 9161.0267527
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3961185, upper bound: 9160.3961185
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3961185, upper bound: 9160.3961185
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.8024421, upper bound: 9160.8024421
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.8024421, upper bound: 9160.8024421
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.8027331, upper bound: 9160.8027331
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.8027331, upper bound: 9160.8027331
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3744077, upper bound: 9160.3740274
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3744307, upper bound: 9160.3740274
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.8728930, upper bound: 9159.8728930
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.8733134, upper bound: 9159.8728930
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3474681, upper bound: 9160.3474681
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3474681, upper bound: 9160.3474681
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.0987842, upper bound: 9160.0983754
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.0983754, upper bound: 9160.0983754
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6023902, upper bound: 9160.6023902
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6023902, upper bound: 9160.6023902
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6024106, upper bound: 9160.6023902
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6025143, upper bound: 9160.6023902
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9157.3136278, upper bound: 9157.3136278
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9157.3136278, upper bound: 9157.3136278
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6341477, upper bound: 9160.6341547
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6341477, upper bound: 9160.6341477
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.5483328, upper bound: 9160.5483328
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.5483328, upper bound: 9160.5483328
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.1032883, upper bound: 9159.1032883
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.1032883, upper bound: 9159.1032883
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.6788187, upper bound: 9159.6788821
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.6788187, upper bound: 9159.6788187
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3443137, upper bound: 9160.3443137
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3443137, upper bound: 9160.3443137
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.2864739, upper bound: 9160.2866588
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.2864739, upper bound: 9160.2867220
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.9549903, upper bound: 9159.9549903
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.9549903, upper bound: 9159.9549903
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9157.6104675, upper bound: 9157.6104675
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9157.6104675, upper bound: 9157.6110378
time: 1.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.8323360, upper bound: 9160.8322185
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.8323414, upper bound: 9160.8322185
time: 0.82 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.82 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9161.6074284, upper bound: 9161.6074284
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9161.6074284, upper bound: 9161.6074284
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9161.6080230, upper bound: 9161.6080230
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9161.6080230, upper bound: 9161.6080230
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9162.6082637, upper bound: 9162.6082637
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9162.6088210, upper bound: 9162.6082637
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9162.5027213, upper bound: 9162.5027213
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9162.5027213, upper bound: 9162.5027213
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9162.1977054, upper bound: 9162.1977054
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9162.1977054, upper bound: 9162.1977054
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9162.1996168, upper bound: 9162.1996168
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9162.1996168, upper bound: 9162.1996168
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9162.0792945, upper bound: 9162.0792945
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9162.0792945, upper bound: 9162.0792945
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9162.0791012, upper bound: 9162.0791012
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9162.0791012, upper bound: 9162.0791012
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9161.6816563, upper bound: 9161.6816563
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9161.6816563, upper bound: 9161.6816563
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9161.2449437, upper bound: 9161.2449437
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9161.2449437, upper bound: 9161.2449437
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9163.1730566, upper bound: 9163.1712227
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9163.1721719, upper bound: 9163.1712227
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9163.7767964, upper bound: 9163.7729104
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9163.7729104, upper bound: 9163.7729104
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9163.7269984, upper bound: 9163.7269984
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9163.7269984, upper bound: 9163.7269984
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9160.3722621, upper bound: 9160.3722621
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9160.3722621, upper bound: 9160.3722621
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9161.8258553, upper bound: 9161.8258553
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9161.8258553, upper bound: 9161.8258553
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9162.6063753, upper bound: 9162.6063753
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9162.6063753, upper bound: 9162.6063753
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9161.9123225, upper bound: 9161.9123225
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9161.9127277, upper bound: 9161.9123225
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9161.9552805, upper bound: 9161.9550259
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9161.9551477, upper bound: 9161.9550259
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9161.4465259, upper bound: 9161.4465259
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9161.4468778, upper bound: 9161.4465259
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9161.5878441, upper bound: 9161.5878441
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9161.5882220, upper bound: 9161.5878441
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9163.0798279, upper bound: 9163.0798279
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9163.0798279, upper bound: 9163.0798279
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9162.2961120, upper bound: 9162.2961120
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9162.2961120, upper bound: 9162.2961120
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9160.5589479, upper bound: 9160.5589479
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9160.5589479, upper bound: 9160.5589479
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9162.4563839, upper bound: 9162.4554320
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9162.4563839, upper bound: 9162.4557531
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9163.8230141, upper bound: 9163.8230141
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9163.8230141, upper bound: 9163.8230141
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9164.0441085, upper bound: 9164.0445693
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9164.0441085, upper bound: 9164.0457270
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9164.0925886, upper bound: 9164.0928381
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9164.0925886, upper bound: 9164.0934543
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9162.5424631, upper bound: 9162.5428021
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9162.5424631, upper bound: 9162.5429301
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9163.7199884, upper bound: 9163.7199884
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9163.7199884, upper bound: 9163.7205328
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9164.1393966, upper bound: 9164.1393966
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9164.1393966, upper bound: 9164.1397404
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9164.1387240, upper bound: 9164.1387240
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9164.1387240, upper bound: 9164.1387240
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9163.6182719, upper bound: 9163.6191574
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9163.6182719, upper bound: 9163.6192019
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9161.0450536, upper bound: 9161.0450496
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9161.0450568, upper bound: 9161.0450496
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9161.0391126, upper bound: 9161.0391126
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9161.0391126, upper bound: 9161.0391126
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9161.0441875, upper bound: 9161.0447157
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9161.0441875, upper bound: 9161.0441875
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9160.7044105, upper bound: 9160.7044105
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9160.7044105, upper bound: 9160.7044105
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9161.0391164, upper bound: 9161.0391126
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9161.0391126, upper bound: 9161.0391126
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9161.0450496, upper bound: 9161.0450496
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9161.0450496, upper bound: 9161.0450496
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9161.0267527, upper bound: 9161.0267527
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9161.0267527, upper bound: 9161.0267527
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9160.3961185, upper bound: 9160.3961185
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9160.3961185, upper bound: 9160.3961185
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9160.8024421, upper bound: 9160.8024421
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9160.8024421, upper bound: 9160.8024421
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9160.8027331, upper bound: 9160.8027331
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9160.8027331, upper bound: 9160.8027331
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9160.3744077, upper bound: 9160.3740274
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9160.3744307, upper bound: 9160.3740274
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9159.8728930, upper bound: 9159.8728930
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9159.8733134, upper bound: 9159.8728930
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9160.3474681, upper bound: 9160.3474681
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9160.3474681, upper bound: 9160.3474681
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9160.0987842, upper bound: 9160.0983754
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9160.0983754, upper bound: 9160.0983754
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9160.6023902, upper bound: 9160.6023902
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9160.6023902, upper bound: 9160.6023902
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9160.6024106, upper bound: 9160.6023902
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9160.6025143, upper bound: 9160.6023902
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 0, lower bound: -9157.3136278, upper bound: 9157.3136278
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 0, lower bound: -9157.3136278, upper bound: 9157.3136278
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9160.6341477, upper bound: 9160.6341547
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9160.6341477, upper bound: 9160.6341477
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9160.5483328, upper bound: 9160.5483328
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9160.5483328, upper bound: 9160.5483328
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9159.1032883, upper bound: 9159.1032883
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9159.1032883, upper bound: 9159.1032883
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9159.6788187, upper bound: 9159.6788821
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9159.6788187, upper bound: 9159.6788187
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9160.3443137, upper bound: 9160.3443137
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9160.3443137, upper bound: 9160.3443137
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9160.2864739, upper bound: 9160.2866588
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9160.2864739, upper bound: 9160.2867220
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9159.9549903, upper bound: 9159.9549903
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9159.9549903, upper bound: 9159.9549903
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 0, lower bound: -9157.6104675, upper bound: 9157.6104675
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.82
Output dim: 0, lower bound: -9157.6104675, upper bound: 9157.6110378
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9160.8323360, upper bound: 9160.8322185
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.82
Output dim: 0, lower bound: -9160.8323414, upper bound: 9160.8322185

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3049964, upper bound: 9160.3049964
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3049964, upper bound: 9160.3049964
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5233329, upper bound: 9161.5233329
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5233329, upper bound: 9161.5233329
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6074284, upper bound: 9161.6074284
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6074284, upper bound: 9161.6074284
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.3660802, upper bound: 9161.3660802
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.3660802, upper bound: 9161.3660802
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.3962051, upper bound: 9162.3962051
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.3962051, upper bound: 9162.3962051
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.5032416, upper bound: 9162.5027213
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.5032566, upper bound: 9162.5027213
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.0727704, upper bound: 9162.0727704
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.0727704, upper bound: 9162.0727704
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.5018794, upper bound: 9162.5018794
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.5018794, upper bound: 9162.5018794
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3049964, upper bound: 9160.3049964
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3049964, upper bound: 9160.3049964
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.1407434, upper bound: 9160.1407434
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.1407434, upper bound: 9160.1407434
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.1418836, upper bound: 9160.1418836
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.1418836, upper bound: 9160.1418836
time: 1.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.1996168, upper bound: 9162.1996168
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.1996168, upper bound: 9162.1996168
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.0792945, upper bound: 9162.0792945
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.0792945, upper bound: 9162.0792945
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.2032410, upper bound: 9160.2032410
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.2032410, upper bound: 9160.2032410
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.0791012, upper bound: 9162.0791012
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.0791012, upper bound: 9162.0791012
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.0791012, upper bound: 9162.0791012
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.0791012, upper bound: 9162.0791012
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1800853, upper bound: 9161.1800853
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1800853, upper bound: 9161.1800853
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.1407434, upper bound: 9160.1407434
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.1407434, upper bound: 9160.1407434
time: 0.85 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 3.40 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.40
Output dim: 0, lower bound: -9160.3049964, upper bound: 9160.3049964
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.40
Output dim: 0, lower bound: -9160.3049964, upper bound: 9160.3049964
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.40
Output dim: 0, lower bound: -9161.5233329, upper bound: 9161.5233329
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.40
Output dim: 0, lower bound: -9161.5233329, upper bound: 9161.5233329
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.40
Output dim: 0, lower bound: -9161.6074284, upper bound: 9161.6074284
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.40
Output dim: 0, lower bound: -9161.6074284, upper bound: 9161.6074284
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.40
Output dim: 0, lower bound: -9161.3660802, upper bound: 9161.3660802
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.40
Output dim: 0, lower bound: -9161.3660802, upper bound: 9161.3660802
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.40
Output dim: 0, lower bound: -9162.3962051, upper bound: 9162.3962051
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.40
Output dim: 0, lower bound: -9162.3962051, upper bound: 9162.3962051
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.40
Output dim: 0, lower bound: -9162.5032416, upper bound: 9162.5027213
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.40
Output dim: 0, lower bound: -9162.5032566, upper bound: 9162.5027213
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.40
Output dim: 0, lower bound: -9162.0727704, upper bound: 9162.0727704
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.40
Output dim: 0, lower bound: -9162.0727704, upper bound: 9162.0727704
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.40
Output dim: 0, lower bound: -9162.5018794, upper bound: 9162.5018794
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.40
Output dim: 0, lower bound: -9162.5018794, upper bound: 9162.5018794
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.40
Output dim: 0, lower bound: -9160.3049964, upper bound: 9160.3049964
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.40
Output dim: 0, lower bound: -9160.3049964, upper bound: 9160.3049964
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.40
Output dim: 0, lower bound: -9160.1407434, upper bound: 9160.1407434
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.40
Output dim: 0, lower bound: -9160.1407434, upper bound: 9160.1407434
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.40
Output dim: 0, lower bound: -9160.1418836, upper bound: 9160.1418836
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.40
Output dim: 0, lower bound: -9160.1418836, upper bound: 9160.1418836
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.40
Output dim: 0, lower bound: -9162.1996168, upper bound: 9162.1996168
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.40
Output dim: 0, lower bound: -9162.1996168, upper bound: 9162.1996168
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.40
Output dim: 0, lower bound: -9162.0792945, upper bound: 9162.0792945
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.40
Output dim: 0, lower bound: -9162.0792945, upper bound: 9162.0792945
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.40
Output dim: 0, lower bound: -9160.2032410, upper bound: 9160.2032410
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.40
Output dim: 0, lower bound: -9160.2032410, upper bound: 9160.2032410
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.40
Output dim: 0, lower bound: -9162.0791012, upper bound: 9162.0791012
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.40
Output dim: 0, lower bound: -9162.0791012, upper bound: 9162.0791012
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.40
Output dim: 0, lower bound: -9162.0791012, upper bound: 9162.0791012
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.40
Output dim: 0, lower bound: -9162.0791012, upper bound: 9162.0791012
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.40
Output dim: 0, lower bound: -9161.1800853, upper bound: 9161.1800853
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.40
Output dim: 0, lower bound: -9161.1800853, upper bound: 9161.1800853
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.40
Output dim: 0, lower bound: -9160.1407434, upper bound: 9160.1407434
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.40
Output dim: 0, lower bound: -9160.1407434, upper bound: 9160.1407434
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9161.2449437, upper bound: 9161.2449437
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9161.2449437, upper bound: 9161.2449437
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9163.1730566, upper bound: 9163.1712227
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9163.1721719, upper bound: 9163.1712227
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9163.7767964, upper bound: 9163.7729104
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9163.7729104, upper bound: 9163.7729104
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9163.7269984, upper bound: 9163.7269984
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9163.7269984, upper bound: 9163.7269984
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9160.3722621, upper bound: 9160.3722621
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9160.3722621, upper bound: 9160.3722621
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9161.8258553, upper bound: 9161.8258553
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9161.8258553, upper bound: 9161.8258553
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9162.6063753, upper bound: 9162.6063753
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9162.6063753, upper bound: 9162.6063753
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9161.9123225, upper bound: 9161.9123225
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9161.9127277, upper bound: 9161.9123225
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9161.9552805, upper bound: 9161.9550259
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9161.9551477, upper bound: 9161.9550259
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9161.4465259, upper bound: 9161.4465259
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9161.4468778, upper bound: 9161.4465259
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9161.5878441, upper bound: 9161.5878441
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9161.5882220, upper bound: 9161.5878441
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9163.0798279, upper bound: 9163.0798279
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9163.0798279, upper bound: 9163.0798279
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9162.2961120, upper bound: 9162.2961120
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9162.2961120, upper bound: 9162.2961120
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9160.5589479, upper bound: 9160.5589479
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9160.5589479, upper bound: 9160.5589479
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9162.4563839, upper bound: 9162.4554320
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9162.4563839, upper bound: 9162.4557531
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9163.8230141, upper bound: 9163.8230141
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9163.8230141, upper bound: 9163.8230141
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9164.0441085, upper bound: 9164.0445693
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9164.0441085, upper bound: 9164.0457270
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9164.0925886, upper bound: 9164.0928381
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9164.0925886, upper bound: 9164.0934543
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9162.5424631, upper bound: 9162.5428021
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9162.5424631, upper bound: 9162.5429301
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9163.7199884, upper bound: 9163.7199884
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9163.7199884, upper bound: 9163.7205328
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9164.1393966, upper bound: 9164.1393966
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9164.1393966, upper bound: 9164.1397404
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9164.1387240, upper bound: 9164.1387240
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9164.1387240, upper bound: 9164.1387240
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9163.6182719, upper bound: 9163.6191574
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9163.6182719, upper bound: 9163.6192019
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9161.0450536, upper bound: 9161.0450496
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9161.0450568, upper bound: 9161.0450496
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9161.0391126, upper bound: 9161.0391126
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9161.0391126, upper bound: 9161.0391126
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9161.0441875, upper bound: 9161.0447157
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9161.0441875, upper bound: 9161.0441875
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9160.7044105, upper bound: 9160.7044105
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9160.7044105, upper bound: 9160.7044105
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9161.0391164, upper bound: 9161.0391126
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9161.0391126, upper bound: 9161.0391126
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9161.0450496, upper bound: 9161.0450496
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9161.0450496, upper bound: 9161.0450496
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9161.0267527, upper bound: 9161.0267527
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9161.0267527, upper bound: 9161.0267527
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9160.3961185, upper bound: 9160.3961185
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9160.3961185, upper bound: 9160.3961185
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9160.8024421, upper bound: 9160.8024421
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9160.8024421, upper bound: 9160.8024421
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9160.8027331, upper bound: 9160.8027331
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9160.8027331, upper bound: 9160.8027331
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9160.3744077, upper bound: 9160.3740274
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9160.3744307, upper bound: 9160.3740274
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9159.8728930, upper bound: 9159.8728930
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9159.8733134, upper bound: 9159.8728930
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9160.3474681, upper bound: 9160.3474681
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9160.3474681, upper bound: 9160.3474681
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9160.0987842, upper bound: 9160.0983754
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9160.0983754, upper bound: 9160.0983754
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9160.6023902, upper bound: 9160.6023902
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9160.6023902, upper bound: 9160.6023902
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9160.6024106, upper bound: 9160.6023902
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9160.6025143, upper bound: 9160.6023902
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9160.6341477, upper bound: 9160.6341547
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9160.6341477, upper bound: 9160.6341477
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9160.5483328, upper bound: 9160.5483328
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9160.5483328, upper bound: 9160.5483328
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9159.1032883, upper bound: 9159.1032883
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9159.1032883, upper bound: 9159.1032883
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9159.6788187, upper bound: 9159.6788821
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9159.6788187, upper bound: 9159.6788187
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9160.3443137, upper bound: 9160.3443137
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9160.3443137, upper bound: 9160.3443137
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9160.2864739, upper bound: 9160.2866588
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9160.2864739, upper bound: 9160.2867220
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9159.9549903, upper bound: 9159.9549903
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9159.9549903, upper bound: 9159.9549903
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9160.8323360, upper bound: 9160.8322185
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -9160.8323414, upper bound: 9160.8322185
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=11463.63671875
rel_dist={0: [-9164.409543284748, 9164.409543284746]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9164.3288204, upper bound: 9164.3288204
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9164.3288204, upper bound: 9164.3294873
time: 0.93 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.74 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.74
Output dim: 0, lower bound: -9164.3288204, upper bound: 9164.3288204
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.74
Output dim: 0, lower bound: -9164.3288204, upper bound: 9164.3294873

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.8895353, upper bound: 9163.8880849
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.8897692, upper bound: 9163.8862218
time: 1.16 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9164.3288190, upper bound: 9164.3283496
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9164.3258676, upper bound: 9164.3294382
time: 0.73 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.53 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.53
Output dim: 0, lower bound: -9163.8895353, upper bound: 9163.8880849
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.53
Output dim: 0, lower bound: -9163.8897692, upper bound: 9163.8862218
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.53
Output dim: 0, lower bound: -9164.3288190, upper bound: 9164.3283496
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.53
Output dim: 0, lower bound: -9164.3258676, upper bound: 9164.3294382

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.4352517, upper bound: 9163.4355850
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.4369689, upper bound: 9163.4351922
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6977932, upper bound: 9161.6974464
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6977932, upper bound: 9161.6974464
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6628958, upper bound: 9162.6637307
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6636306, upper bound: 9162.6637307
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.8822953, upper bound: 9163.8837488
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.8818546, upper bound: 9163.8828828
time: 0.83 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.75 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.75
Output dim: 0, lower bound: -9163.4352517, upper bound: 9163.4355850
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.75
Output dim: 0, lower bound: -9163.4369689, upper bound: 9163.4351922
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.75
Output dim: 0, lower bound: -9161.6977932, upper bound: 9161.6974464
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.75
Output dim: 0, lower bound: -9161.6977932, upper bound: 9161.6974464
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.75
Output dim: 0, lower bound: -9162.6628958, upper bound: 9162.6637307
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.75
Output dim: 0, lower bound: -9162.6636306, upper bound: 9162.6637307
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.75
Output dim: 0, lower bound: -9163.8822953, upper bound: 9163.8837488
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.75
Output dim: 0, lower bound: -9163.8818546, upper bound: 9163.8828828

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.3589355, upper bound: 9163.3603637
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.3601796, upper bound: 9163.3605276
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.4333798, upper bound: 9163.4347405
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.4350286, upper bound: 9163.4342378
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6219795, upper bound: 9161.6219795
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6223026, upper bound: 9161.6219795
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6976825, upper bound: 9161.6973099
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6973099, upper bound: 9161.6973099
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.3888994, upper bound: 9162.3891304
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.3889287, upper bound: 9162.3881868
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6363166, upper bound: 9160.6363166
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6363166, upper bound: 9160.6363166
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.6285067, upper bound: 9163.6303191
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.6282740, upper bound: 9163.6299964
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.7490584, upper bound: 9163.7495268
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.7490584, upper bound: 9163.7490584
time: 5.11 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 7.38 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.38
Output dim: 0, lower bound: -9163.3589355, upper bound: 9163.3603637
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.38
Output dim: 0, lower bound: -9163.3601796, upper bound: 9163.3605276
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.38
Output dim: 0, lower bound: -9163.4333798, upper bound: 9163.4347405
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.38
Output dim: 0, lower bound: -9163.4350286, upper bound: 9163.4342378
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.38
Output dim: 0, lower bound: -9161.6219795, upper bound: 9161.6219795
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.38
Output dim: 0, lower bound: -9161.6223026, upper bound: 9161.6219795
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.38
Output dim: 0, lower bound: -9161.6976825, upper bound: 9161.6973099
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.38
Output dim: 0, lower bound: -9161.6973099, upper bound: 9161.6973099
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.38
Output dim: 0, lower bound: -9162.3888994, upper bound: 9162.3891304
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.38
Output dim: 0, lower bound: -9162.3889287, upper bound: 9162.3881868
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.38
Output dim: 0, lower bound: -9160.6363166, upper bound: 9160.6363166
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.38
Output dim: 0, lower bound: -9160.6363166, upper bound: 9160.6363166
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.38
Output dim: 0, lower bound: -9163.6285067, upper bound: 9163.6303191
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.38
Output dim: 0, lower bound: -9163.6282740, upper bound: 9163.6299964
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.38
Output dim: 0, lower bound: -9163.7490584, upper bound: 9163.7495268
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.38
Output dim: 0, lower bound: -9163.7490584, upper bound: 9163.7490584

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.3589355, upper bound: 9163.3600087
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.3589355, upper bound: 9163.3603637
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.3597784, upper bound: 9163.3599833
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.3590289, upper bound: 9163.3600714
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.4333798, upper bound: 9163.4333798
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.4333798, upper bound: 9163.4347405
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.9203399, upper bound: 9162.9210868
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.9217742, upper bound: 9162.9210822
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6219795, upper bound: 9161.6219795
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6219795, upper bound: 9161.6219795
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.3671730, upper bound: 9161.3669445
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.3669445, upper bound: 9161.3669445
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6976825, upper bound: 9161.6973099
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6976401, upper bound: 9161.6973099
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6941896, upper bound: 9161.6941896
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6941896, upper bound: 9161.6941896
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.3888994, upper bound: 9162.3891304
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.3887706, upper bound: 9162.3881868
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.3889287, upper bound: 9162.3881868
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.3881868, upper bound: 9162.3881868
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.4573135, upper bound: 9160.4573135
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.4573135, upper bound: 9160.4573135
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.5661141, upper bound: 9160.5661141
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.5661141, upper bound: 9160.5661141
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.9467424, upper bound: 9162.9481143
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.9467424, upper bound: 9162.9488955
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.2768658, upper bound: 9160.2776076
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.2768658, upper bound: 9160.2770895
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.7032665, upper bound: 9163.7033438
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.7032665, upper bound: 9163.7033438
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.7490584, upper bound: 9163.7490584
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.7490584, upper bound: 9163.7490584
time: 0.69 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.82 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 0, lower bound: -9163.3589355, upper bound: 9163.3600087
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 0, lower bound: -9163.3589355, upper bound: 9163.3603637
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 0, lower bound: -9163.3597784, upper bound: 9163.3599833
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 0, lower bound: -9163.3590289, upper bound: 9163.3600714
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 0, lower bound: -9163.4333798, upper bound: 9163.4333798
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 0, lower bound: -9163.4333798, upper bound: 9163.4347405
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 0, lower bound: -9162.9203399, upper bound: 9162.9210868
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 0, lower bound: -9162.9217742, upper bound: 9162.9210822
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 0, lower bound: -9161.6219795, upper bound: 9161.6219795
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 0, lower bound: -9161.6219795, upper bound: 9161.6219795
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 0, lower bound: -9161.3671730, upper bound: 9161.3669445
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 0, lower bound: -9161.3669445, upper bound: 9161.3669445
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 0, lower bound: -9161.6976825, upper bound: 9161.6973099
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 0, lower bound: -9161.6976401, upper bound: 9161.6973099
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 0, lower bound: -9161.6941896, upper bound: 9161.6941896
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 0, lower bound: -9161.6941896, upper bound: 9161.6941896
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 0, lower bound: -9162.3888994, upper bound: 9162.3891304
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 0, lower bound: -9162.3887706, upper bound: 9162.3881868
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 0, lower bound: -9162.3889287, upper bound: 9162.3881868
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 0, lower bound: -9162.3881868, upper bound: 9162.3881868
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 0, lower bound: -9160.4573135, upper bound: 9160.4573135
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 0, lower bound: -9160.4573135, upper bound: 9160.4573135
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 0, lower bound: -9160.5661141, upper bound: 9160.5661141
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 0, lower bound: -9160.5661141, upper bound: 9160.5661141
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 0, lower bound: -9162.9467424, upper bound: 9162.9481143
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 0, lower bound: -9162.9467424, upper bound: 9162.9488955
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 0, lower bound: -9160.2768658, upper bound: 9160.2776076
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 0, lower bound: -9160.2768658, upper bound: 9160.2770895
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 0, lower bound: -9163.7032665, upper bound: 9163.7033438
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 0, lower bound: -9163.7032665, upper bound: 9163.7033438
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 0, lower bound: -9163.7490584, upper bound: 9163.7490584
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.82
Output dim: 0, lower bound: -9163.7490584, upper bound: 9163.7490584

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.3589355, upper bound: 9163.3589355
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.3589355, upper bound: 9163.3600045
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.3587201, upper bound: 9163.3587201
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.3587201, upper bound: 9163.3601689
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.1630353, upper bound: 9163.1633719
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.1618595, upper bound: 9163.1643288
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.4307460, upper bound: 9162.4307460
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.4307460, upper bound: 9162.4311720
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.3584444, upper bound: 9163.3584444
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.3584444, upper bound: 9163.3584444
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.4333798, upper bound: 9163.4345712
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.4333798, upper bound: 9163.4347405
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.9203399, upper bound: 9162.9210666
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.9203399, upper bound: 9162.9210666
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6735729, upper bound: 9162.6744512
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6735729, upper bound: 9162.6744393
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3360788, upper bound: 9160.3360788
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3360788, upper bound: 9160.3360788
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6209311, upper bound: 9161.6209311
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6209311, upper bound: 9161.6209311
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.8238512, upper bound: 9160.8238512
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.8239059, upper bound: 9160.8238512
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.3665480, upper bound: 9161.3659683
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.3659683, upper bound: 9161.3659683
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6218666, upper bound: 9161.6218666
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6221978, upper bound: 9161.6218666
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6807730, upper bound: 9161.6807730
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6811540, upper bound: 9161.6807730
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.2778847, upper bound: 9161.2778847
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.2778847, upper bound: 9161.2778847
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6930734, upper bound: 9161.6930734
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6930734, upper bound: 9161.6930734
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.3037700, upper bound: 9162.3047234
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.3043683, upper bound: 9162.3036565
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.2916295, upper bound: 9162.2906872
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.2906872, upper bound: 9162.2906872
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1661640, upper bound: 9161.1661640
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.1661640, upper bound: 9161.1661640
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.3881868, upper bound: 9162.3881868
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.3881868, upper bound: 9162.3881868
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.4573135, upper bound: 9160.4573135
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.4573135, upper bound: 9160.4573135
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.4570734, upper bound: 9160.4570734
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.4570734, upper bound: 9160.4570734
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.5661141, upper bound: 9160.5661141
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.5661141, upper bound: 9160.5661141
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.5658292, upper bound: 9160.5658292
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.5658292, upper bound: 9160.5658292
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.9446720, upper bound: 9162.9446720
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.9446720, upper bound: 9162.9446720
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.3685903, upper bound: 9162.3703660
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.3685903, upper bound: 9162.3703211
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.2768658, upper bound: 9160.2776076
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.2768658, upper bound: 9160.2774815
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.2683141, upper bound: 9160.2685378
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.2683141, upper bound: 9160.2684362
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.7028212, upper bound: 9163.7028477
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.7028212, upper bound: 9163.7028212
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.7032665, upper bound: 9163.7033258
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.7032665, upper bound: 9163.7033438
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.0952189, upper bound: 9163.0952189
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.0952189, upper bound: 9163.0952189
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.0952189, upper bound: 9163.0952189
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.0952189, upper bound: 9163.0952189
time: 0.84 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.28 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9163.3589355, upper bound: 9163.3589355
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9163.3589355, upper bound: 9163.3600045
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9163.3587201, upper bound: 9163.3587201
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9163.3587201, upper bound: 9163.3601689
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9163.1630353, upper bound: 9163.1633719
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9163.1618595, upper bound: 9163.1643288
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9162.4307460, upper bound: 9162.4307460
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9162.4307460, upper bound: 9162.4311720
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9163.3584444, upper bound: 9163.3584444
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9163.3584444, upper bound: 9163.3584444
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9163.4333798, upper bound: 9163.4345712
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9163.4333798, upper bound: 9163.4347405
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9162.9203399, upper bound: 9162.9210666
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9162.9203399, upper bound: 9162.9210666
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9162.6735729, upper bound: 9162.6744512
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9162.6735729, upper bound: 9162.6744393
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9160.3360788, upper bound: 9160.3360788
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9160.3360788, upper bound: 9160.3360788
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9161.6209311, upper bound: 9161.6209311
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9161.6209311, upper bound: 9161.6209311
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9160.8238512, upper bound: 9160.8238512
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9160.8239059, upper bound: 9160.8238512
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9161.3665480, upper bound: 9161.3659683
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9161.3659683, upper bound: 9161.3659683
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9161.6218666, upper bound: 9161.6218666
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9161.6221978, upper bound: 9161.6218666
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9161.6807730, upper bound: 9161.6807730
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9161.6811540, upper bound: 9161.6807730
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9161.2778847, upper bound: 9161.2778847
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9161.2778847, upper bound: 9161.2778847
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9161.6930734, upper bound: 9161.6930734
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9161.6930734, upper bound: 9161.6930734
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9162.3037700, upper bound: 9162.3047234
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9162.3043683, upper bound: 9162.3036565
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9162.2916295, upper bound: 9162.2906872
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9162.2906872, upper bound: 9162.2906872
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9161.1661640, upper bound: 9161.1661640
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9161.1661640, upper bound: 9161.1661640
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9162.3881868, upper bound: 9162.3881868
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9162.3881868, upper bound: 9162.3881868
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9160.4573135, upper bound: 9160.4573135
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9160.4573135, upper bound: 9160.4573135
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9160.4570734, upper bound: 9160.4570734
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9160.4570734, upper bound: 9160.4570734
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9160.5661141, upper bound: 9160.5661141
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9160.5661141, upper bound: 9160.5661141
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9160.5658292, upper bound: 9160.5658292
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9160.5658292, upper bound: 9160.5658292
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9162.9446720, upper bound: 9162.9446720
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9162.9446720, upper bound: 9162.9446720
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9162.3685903, upper bound: 9162.3703660
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9162.3685903, upper bound: 9162.3703211
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9160.2768658, upper bound: 9160.2776076
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9160.2768658, upper bound: 9160.2774815
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9160.2683141, upper bound: 9160.2685378
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9160.2683141, upper bound: 9160.2684362
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9163.7028212, upper bound: 9163.7028477
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9163.7028212, upper bound: 9163.7028212
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9163.7032665, upper bound: 9163.7033258
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9163.7032665, upper bound: 9163.7033438
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9163.0952189, upper bound: 9163.0952189
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9163.0952189, upper bound: 9163.0952189
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9163.0952189, upper bound: 9163.0952189
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -9163.0952189, upper bound: 9163.0952189

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.3587201, upper bound: 9163.3587201
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.3587201, upper bound: 9163.3587201
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.3038159, upper bound: 9163.3038159
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.3038159, upper bound: 9163.3038159
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.3587201, upper bound: 9163.3587201
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.3587201, upper bound: 9163.3587201
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.3587201, upper bound: 9163.3601689
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.3587201, upper bound: 9163.3587201
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.1163428, upper bound: 9163.1177561
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.1163428, upper bound: 9163.1177561
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.0386583, upper bound: 9163.0418960
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.0386583, upper bound: 9163.0388063
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.4307460, upper bound: 9162.4307460
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.4307460, upper bound: 9162.4307460
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.3787051, upper bound: 9162.3788291
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.3787051, upper bound: 9162.3787051
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.3032696, upper bound: 9163.3032696
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.3032696, upper bound: 9163.3032696
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.3580486, upper bound: 9163.3580486
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.3580486, upper bound: 9163.3580486
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.4331829, upper bound: 9163.4333190
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.4331829, upper bound: 9163.4343919
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.9203399, upper bound: 9162.9214542
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.9203399, upper bound: 9162.9214769
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.9203399, upper bound: 9162.9210666
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.9203399, upper bound: 9162.9203399
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.8688004, upper bound: 9162.8688004
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.8688004, upper bound: 9162.8688004
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6735729, upper bound: 9162.6738697
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6735729, upper bound: 9162.6744512
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6735729, upper bound: 9162.6744393
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6735729, upper bound: 9162.6735729
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.2199949, upper bound: 9160.2199949
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.2199949, upper bound: 9160.2199949
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.2199949, upper bound: 9160.2199949
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.2199949, upper bound: 9160.2199949
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3358277, upper bound: 9160.3358277
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3358277, upper bound: 9160.3358277
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3358277, upper bound: 9160.3358277
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.3358277, upper bound: 9160.3358277
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.8232030, upper bound: 9160.8232030
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.8232030, upper bound: 9160.8232030
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.8238444, upper bound: 9160.8237954
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.8237954, upper bound: 9160.8237954
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.8231442, upper bound: 9160.8231442
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.8231597, upper bound: 9160.8231442
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.3658321, upper bound: 9161.3658321
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.3658321, upper bound: 9161.3658321
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5978771, upper bound: 9161.5978771
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5978771, upper bound: 9161.5978771
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6218292, upper bound: 9161.6218292
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6218292, upper bound: 9161.6218292
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6462428, upper bound: 9161.6462428
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6462428, upper bound: 9161.6462428
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.3480079, upper bound: 9161.3480079
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.3482806, upper bound: 9161.3480079
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.2778847, upper bound: 9161.2778847
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.2778847, upper bound: 9161.2778847
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.2346537, upper bound: 9161.2346537
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.2346537, upper bound: 9161.2346537
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6930734, upper bound: 9161.6930734
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6930734, upper bound: 9161.6930734
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6116807, upper bound: 9161.6116807
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6116807, upper bound: 9161.6116807
time: 0.76 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.73 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9163.3587201, upper bound: 9163.3587201
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9163.3587201, upper bound: 9163.3587201
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9163.3038159, upper bound: 9163.3038159
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9163.3038159, upper bound: 9163.3038159
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9163.3587201, upper bound: 9163.3587201
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9163.3587201, upper bound: 9163.3587201
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9163.3587201, upper bound: 9163.3601689
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9163.3587201, upper bound: 9163.3587201
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9163.1163428, upper bound: 9163.1177561
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9163.1163428, upper bound: 9163.1177561
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9163.0386583, upper bound: 9163.0418960
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9163.0386583, upper bound: 9163.0388063
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9162.4307460, upper bound: 9162.4307460
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9162.4307460, upper bound: 9162.4307460
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9162.3787051, upper bound: 9162.3788291
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9162.3787051, upper bound: 9162.3787051
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9163.3032696, upper bound: 9163.3032696
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9163.3032696, upper bound: 9163.3032696
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9163.3580486, upper bound: 9163.3580486
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9163.3580486, upper bound: 9163.3580486
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9163.4331829, upper bound: 9163.4333190
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9163.4331829, upper bound: 9163.4343919
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9162.9203399, upper bound: 9162.9214542
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9162.9203399, upper bound: 9162.9214769
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9162.9203399, upper bound: 9162.9210666
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9162.9203399, upper bound: 9162.9203399
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9162.8688004, upper bound: 9162.8688004
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9162.8688004, upper bound: 9162.8688004
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9162.6735729, upper bound: 9162.6738697
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9162.6735729, upper bound: 9162.6744512
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9162.6735729, upper bound: 9162.6744393
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9162.6735729, upper bound: 9162.6735729
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9160.2199949, upper bound: 9160.2199949
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9160.2199949, upper bound: 9160.2199949
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9160.2199949, upper bound: 9160.2199949
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9160.2199949, upper bound: 9160.2199949
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9160.3358277, upper bound: 9160.3358277
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9160.3358277, upper bound: 9160.3358277
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9160.3358277, upper bound: 9160.3358277
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9160.3358277, upper bound: 9160.3358277
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9160.8232030, upper bound: 9160.8232030
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9160.8232030, upper bound: 9160.8232030
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9160.8238444, upper bound: 9160.8237954
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9160.8237954, upper bound: 9160.8237954
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9160.8231442, upper bound: 9160.8231442
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9160.8231597, upper bound: 9160.8231442
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9161.3658321, upper bound: 9161.3658321
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9161.3658321, upper bound: 9161.3658321
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9161.5978771, upper bound: 9161.5978771
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9161.5978771, upper bound: 9161.5978771
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9161.6218292, upper bound: 9161.6218292
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9161.6218292, upper bound: 9161.6218292
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9161.6462428, upper bound: 9161.6462428
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9161.6462428, upper bound: 9161.6462428
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9161.3480079, upper bound: 9161.3480079
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9161.3482806, upper bound: 9161.3480079
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9161.2778847, upper bound: 9161.2778847
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9161.2778847, upper bound: 9161.2778847
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9161.2346537, upper bound: 9161.2346537
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9161.2346537, upper bound: 9161.2346537
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9161.6930734, upper bound: 9161.6930734
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9161.6930734, upper bound: 9161.6930734
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9161.6116807, upper bound: 9161.6116807
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -9161.6116807, upper bound: 9161.6116807
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 0, lower bound: -9162.3037700, upper bound: 9162.3047234
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 0, lower bound: -9162.3043683, upper bound: 9162.3036565
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 0, lower bound: -9162.2916295, upper bound: 9162.2906872
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 0, lower bound: -9162.2906872, upper bound: 9162.2906872
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 0, lower bound: -9161.1661640, upper bound: 9161.1661640
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 0, lower bound: -9161.1661640, upper bound: 9161.1661640
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 0, lower bound: -9162.3881868, upper bound: 9162.3881868
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 0, lower bound: -9162.3881868, upper bound: 9162.3881868
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 0, lower bound: -9160.4573135, upper bound: 9160.4573135
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 0, lower bound: -9160.4573135, upper bound: 9160.4573135
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 0, lower bound: -9160.4570734, upper bound: 9160.4570734
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 0, lower bound: -9160.4570734, upper bound: 9160.4570734
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 0, lower bound: -9160.5661141, upper bound: 9160.5661141
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 0, lower bound: -9160.5661141, upper bound: 9160.5661141
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 0, lower bound: -9160.5658292, upper bound: 9160.5658292
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 0, lower bound: -9160.5658292, upper bound: 9160.5658292
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 0, lower bound: -9162.9446720, upper bound: 9162.9446720
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 0, lower bound: -9162.9446720, upper bound: 9162.9446720
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 0, lower bound: -9162.3685903, upper bound: 9162.3703660
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 0, lower bound: -9162.3685903, upper bound: 9162.3703211
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 0, lower bound: -9160.2768658, upper bound: 9160.2776076
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 0, lower bound: -9160.2768658, upper bound: 9160.2774815
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 0, lower bound: -9160.2683141, upper bound: 9160.2685378
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 0, lower bound: -9160.2683141, upper bound: 9160.2684362
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 0, lower bound: -9163.7028212, upper bound: 9163.7028477
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 0, lower bound: -9163.7028212, upper bound: 9163.7028212
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 0, lower bound: -9163.7032665, upper bound: 9163.7033258
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 0, lower bound: -9163.7032665, upper bound: 9163.7033438
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 0, lower bound: -9163.0952189, upper bound: 9163.0952189
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 0, lower bound: -9163.0952189, upper bound: 9163.0952189
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 0, lower bound: -9163.0952189, upper bound: 9163.0952189
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.73
Output dim: 0, lower bound: -9163.0952189, upper bound: 9163.0952189
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=11463.63671875
rel_dist={0: [-9164.405028251073, 9164.405028251073]}

## Binary Search with RS_random_Z Result
status: None
Maximum delta epsilon: None
execution time: 1129.59 seconds
