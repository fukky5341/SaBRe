## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_2.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 5471.923754115261


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344)
1: (-1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438)
2: (-1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969)
3: (-1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930)
4: (-1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016)

## BASE Result
execution time: IAR + LP analysis = 1.95 + 2.41 = 4.36 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -5471.9793324, upper bound: 5471.9793324


# Binary Search by BASE starts (time budget: 1195.64 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=6162.86279296875
rel_dist={3: [-5471.979332446701, 5471.979332446701]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=6162.86279296875
rel_dist={3: [-5471.979020840545, 5471.979020840547]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=6162.86279296875
rel_dist={3: [-5471.978598948089, 5471.97859894809]}

## Binary search (step 3) starts
Candidate diff: 0.0625000


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=6162.86279296875
rel_dist={3: [-5471.978217283832, 5471.978217283833]}

## Binary search (step 4) starts
Candidate diff: 0.0312500


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=6162.86279296875
rel_dist={3: [-5471.977887691975, 5471.977887691801]}

## Binary search (step 5) starts
Candidate diff: 0.0156250


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=6162.86279296875
rel_dist={3: [-5471.9776877308705, 5471.977687730634]}

## Binary search (step 6) starts
Candidate diff: 0.0078125


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0078125, mid=0.0078125, abs_max=6162.86279296875
rel_dist={3: [-5471.977546026561, 5471.977546026394]}

## Binary search (step 7) starts
Candidate diff: 0.0039062


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0039062, mid=0.0039062, abs_max=6162.86279296875
rel_dist={3: [-5471.977467799268, 5471.977467799357]}

## Binary search (step 8) starts
Candidate diff: 0.0019531


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0019531, mid=0.0019531, abs_max=6162.86279296875
rel_dist={3: [-5471.977426835224, 5471.977426835225]}

## Binary search (step 9) starts
Candidate diff: 0.0009766


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0009766, mid=0.0009766, abs_max=6162.86279296875
rel_dist={3: [-5471.977406353012, 5471.977406353071]}

## Binary search (step 10) starts
Candidate diff: 0.0004883


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0004883, mid=0.0004883, abs_max=6162.86279296875
rel_dist={3: [-5471.977396111982, 5471.977396112639]}

## Binary search (step 11) starts
Candidate diff: 0.0002441


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0002441, mid=0.0002441, abs_max=6162.86279296875
rel_dist={3: [-5471.977390991502, 5471.9773909915275]}

## Binary search (step 12) starts
Candidate diff: 0.0001221


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0001221, mid=0.0001221, abs_max=6162.86279296875
rel_dist={3: [-5471.977388431275, 5471.977388431009]}

## Binary search (step 13) starts
Candidate diff: 0.0000610


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000610, mid=0.0000610, abs_max=6162.86279296875
rel_dist={3: [-5471.9773871511625, 5471.977387150775]}

## Binary search (step 14) starts
Candidate diff: 0.0000305


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000305, mid=0.0000305, abs_max=6162.86279296875
rel_dist={3: [-5471.977386511122, 5471.977386510684]}

## Binary search (step 15) starts
Candidate diff: 0.0000153


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000153, mid=0.0000153, abs_max=6162.86279296875
rel_dist={3: [-5471.977386191007, 5471.977386190945]}

## Binary search (step 16) starts
Candidate diff: 0.0000076


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000076, mid=0.0000076, abs_max=6162.86279296875
rel_dist={3: [-5471.977386031454, 5471.977386031294]}

## Binary search (step 17) starts
Candidate diff: 0.0000038


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000038, mid=0.0000038, abs_max=6162.86279296875
rel_dist={3: [-5471.977385951257, 5471.97738595117]}

## Binary search (step 18) starts
Candidate diff: 0.0000019


## IAR start
Binary search (step 18): status=Status.UNKNOWN, low=0.0000000, high=0.0000019, mid=0.0000019, abs_max=6162.86279296875
rel_dist={3: [-5471.977385911711, 5471.977385912429]}

## Binary search (step 19) starts
Candidate diff: 0.0000010


## IAR start
Binary search (step 19): status=Status.UNKNOWN, low=0.0000000, high=0.0000010, mid=0.0000010, abs_max=6162.86279296875
rel_dist={3: [-5471.977385894648, 5471.9773858932]}

## Binary Search Result
Binary search time: 88.63 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 1107.00 seconds

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9791571, upper bound: 5471.9792311
time: 0.91 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9792164, upper bound: 5471.9792164
time: 0.92 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.00 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.00
Output dim: 3, lower bound: -5471.9791571, upper bound: 5471.9792311
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.00
Output dim: 3, lower bound: -5471.9792164, upper bound: 5471.9792164

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -850.7169800, 3697.8376465, -863.1370239, 3753.3532715, -4604.0693359, 4560.9746094
1: -1071.5225830, 4196.7158203, -1086.8886719, 4259.3261719, -5330.8486328, 5283.6035156
2: -1089.2393799, 4190.8291016, -1105.0872803, 4253.2548828, -5342.4931641, 5295.9160156
3: -1711.6445312, 4361.7553711, -1736.3594971, 4426.5043945, -6138.1489258, 6098.1137695
4: -1716.2598877, 4177.3193359, -1741.8470459, 4238.6694336, -5954.9291992, 5919.1665039

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783673, upper bound: 5471.9792309
time: 0.84 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783670, upper bound: 5471.9785007
time: 0.90 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -904.5838623, 3936.6628418, -862.9957886, 3752.7248535, -4657.3085938, 4799.6577148
1: -1139.0826416, 4467.9472656, -1086.7114258, 4258.6132812, -5397.6953125, 5554.6577148
2: -1157.8500977, 4461.1088867, -1104.9068604, 4252.5478516, -5410.3979492, 5566.0156250
3: -1819.3218994, 4642.7978516, -1736.0731201, 4425.7656250, -6245.0869141, 6378.8706055
4: -1825.1307373, 4443.3935547, -1741.5571289, 4237.9697266, -6063.1005859, 6184.9506836

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9792161, upper bound: 5471.9784862
time: 0.87 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784859, upper bound: 5471.9784859
time: 0.97 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.77 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.77
Output dim: 3, lower bound: -5471.9783673, upper bound: 5471.9792309
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.77
Output dim: 3, lower bound: -5471.9783670, upper bound: 5471.9785007
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 3.77
Output dim: 3, lower bound: -5471.9792161, upper bound: 5471.9784862
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 3.77
Output dim: 3, lower bound: -5471.9784859, upper bound: 5471.9784859

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -850.7169800, 3697.8376465, -822.2101440, 3580.1765137, -4430.8930664, 4520.0478516
1: -1071.5225830, 4196.7158203, -1034.9815674, 4063.3266602, -5134.8491211, 5231.6962891
2: -1089.2393799, 4190.8291016, -1052.5050049, 4056.0151367, -5145.2543945, 5243.3334961
3: -1711.6445312, 4361.7553711, -1654.7614746, 4222.3891602, -5934.0336914, 6016.5166016
4: -1716.2598877, 4177.3193359, -1659.9338379, 4042.0654297, -5758.3251953, 5837.2529297

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783670, upper bound: 5471.9785007
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783670, upper bound: 5471.9785007
time: 1.05 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -845.4605713, 3676.2712402, -900.7853394, 3941.4938965, -4786.9545898, 4577.0566406
1: -1064.9014893, 4172.1215820, -1134.5142822, 4472.5200195, -5537.4213867, 5306.6357422
2: -1082.4782715, 4166.3183594, -1153.3151855, 4465.1274414, -5547.6054688, 5319.6337891
3: -1701.2783203, 4336.0166016, -1816.7711182, 4644.8984375, -6346.1767578, 6152.7875977
4: -1705.7464600, 4152.6083984, -1819.2148438, 4447.2148438, -6152.9609375, 5971.8227539

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783670, upper bound: 5471.9785007
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783670, upper bound: 5471.9785007
time: 0.95 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -863.0641479, 3760.9191895, -862.9957886, 3752.7248535, -4615.7890625, 4623.9135742
1: -1086.4407959, 4269.1186523, -1086.7114258, 4258.6132812, -5345.0541992, 5355.8300781
2: -1104.5764160, 4261.0634766, -1104.9068604, 4252.5478516, -5357.1235352, 5365.9697266
3: -1736.5150146, 4435.6914062, -1736.0731201, 4425.7656250, -6162.2797852, 6171.7636719
4: -1742.0505371, 4244.0839844, -1741.5571289, 4237.9697266, -5980.0205078, 5985.6411133

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784859, upper bound: 5471.9784859
time: 0.87 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784859, upper bound: 5471.9784859
time: 1.03 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -943.1829224, 4128.3657227, -857.7442627, 3731.1689453, -4674.3515625, 4986.1088867
1: -1187.9083252, 4685.2797852, -1080.0954590, 4234.0273438, -5421.9355469, 5765.3745117
2: -1207.4289551, 4676.9970703, -1098.1505127, 4228.0473633, -5435.4765625, 5775.1474609
3: -1901.7006836, 4865.4750977, -1725.7155762, 4400.0366211, -6301.7373047, 6591.1904297
4: -1904.5893555, 4656.3349609, -1731.0509033, 4213.2656250, -6117.8549805, 6387.3847656

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784859, upper bound: 5471.9784859
time: 1.13 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784859, upper bound: 5471.9784859
time: 1.09 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.19 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.19
Output dim: 3, lower bound: -5471.9783670, upper bound: 5471.9785007
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.19
Output dim: 3, lower bound: -5471.9783670, upper bound: 5471.9785007
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.19
Output dim: 3, lower bound: -5471.9783670, upper bound: 5471.9785007
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.19
Output dim: 3, lower bound: -5471.9783670, upper bound: 5471.9785007
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 4.19
Output dim: 3, lower bound: -5471.9784859, upper bound: 5471.9784859
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 4.19
Output dim: 3, lower bound: -5471.9784859, upper bound: 5471.9784859
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 4.19
Output dim: 3, lower bound: -5471.9784859, upper bound: 5471.9784859
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 4.19
Output dim: 3, lower bound: -5471.9784859, upper bound: 5471.9784859

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -809.6653442, 3524.1467285, -822.2101440, 3580.1765137, -4389.8417969, 4346.3569336
1: -1019.4607544, 4000.1494141, -1034.9815674, 4063.3266602, -5082.7875977, 5035.1308594
2: -1036.5018311, 3992.9978027, -1052.5050049, 4056.0151367, -5092.5170898, 5045.5019531
3: -1629.8031006, 4157.0307617, -1654.7614746, 4222.3891602, -5852.1918945, 5811.7919922
4: -1634.1029053, 3980.1218262, -1659.9338379, 4042.0654297, -5676.1684570, 5640.0551758

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781246, upper bound: 5471.9792262
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783662, upper bound: 5471.9792258
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -888.3382568, 3885.9843750, -822.2101440, 3580.1765137, -4468.5146484, 4708.1943359
1: -1119.1064453, 4409.9409180, -1034.9815674, 4063.3266602, -5182.4331055, 5444.9213867
2: -1137.4340820, 4402.6269531, -1052.5050049, 4056.0151367, -5193.4492188, 5455.1313477
3: -1792.0124512, 4580.1577148, -1654.7614746, 4222.3891602, -6014.4013672, 6234.9189453
4: -1793.5903320, 4385.8295898, -1659.9338379, 4042.0654297, -5835.6557617, 6045.7636719

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 38

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781246, upper bound: 5471.9792262
time: 1.10 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783662, upper bound: 5471.9792258
time: 1.08 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -809.6653442, 3524.1467285, -900.7853394, 3941.4938965, -4751.1591797, 4424.9321289
1: -1019.4607544, 4000.1494141, -1134.5142822, 4472.5200195, -5491.9809570, 5134.6635742
2: -1036.5018311, 3992.9978027, -1153.3151855, 4465.1274414, -5501.6293945, 5146.3125000
3: -1629.8031006, 4157.0307617, -1816.7711182, 4644.8984375, -6274.7016602, 5973.8017578
4: -1634.1029053, 3980.1218262, -1819.2148438, 4447.2148438, -6081.3178711, 5799.3354492

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9763551, upper bound: 5471.9771959
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783658, upper bound: 5471.9784960
time: 1.09 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -888.3382568, 3885.9843750, -900.7853394, 3941.4938965, -4829.8320312, 4786.7695312
1: -1119.1064453, 4409.9409180, -1134.5142822, 4472.5200195, -5591.6264648, 5544.4545898
2: -1137.4340820, 4402.6269531, -1153.3151855, 4465.1274414, -5602.5615234, 5555.9423828
3: -1792.0124512, 4580.1577148, -1816.7711182, 4644.8984375, -6436.9106445, 6396.9287109
4: -1793.5903320, 4385.8295898, -1819.2148438, 4447.2148438, -6240.8046875, 6205.0444336

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9770828, upper bound: 5471.9767111
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783658, upper bound: 5471.9784960
time: 1.18 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: -863.0641479, 3760.9191895, -822.0680542, 3579.5488281, -4442.6127930, 4582.9868164
1: -1086.4407959, 4269.1186523, -1034.8034668, 4062.6137695, -5149.0546875, 5303.9218750
2: -1104.5764160, 4261.0634766, -1052.3231201, 4055.3093262, -5159.8852539, 5313.3857422
3: -1736.5150146, 4435.6914062, -1654.4739990, 4221.6499023, -5958.1640625, 6090.1645508
4: -1742.0505371, 4244.0839844, -1659.6434326, 4041.3674316, -5783.4174805, 5903.7275391

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_A1_B1_A1

### Relational analysis result of IS_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9792149, upper bound: 5471.9782435
time: 1.23 seconds

## Relational analysis of IS_A2_A1_B1_A2

### Relational analysis result of IS_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9792145, upper bound: 5471.9784851
time: 1.13 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: -863.0641479, 3760.9191895, -900.6458740, 3940.8784180, -4803.9423828, 4661.5644531
1: -1086.4407959, 4269.1186523, -1134.3397217, 4471.8212891, -5558.2622070, 5403.4584961
2: -1104.5764160, 4261.0634766, -1153.1384277, 4464.4355469, -5569.0117188, 5414.2011719
3: -1736.5150146, 4435.6914062, -1816.4885254, 4644.1738281, -6380.6889648, 6252.1796875
4: -1742.0505371, 4244.0839844, -1818.9307861, 4446.5278320, -6188.5781250, 6063.0146484

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_A1_B2_A1

### Relational analysis result of IS_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9792149, upper bound: 5471.9782435
time: 0.92 seconds

## Relational analysis of IS_A2_A1_B2_A2

### Relational analysis result of IS_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9792145, upper bound: 5471.9784851
time: 0.98 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -943.1829224, 4128.3657227, -822.0680542, 3579.5488281, -4522.7319336, 4950.4335938
1: -1187.9083252, 4685.2797852, -1034.8034668, 4062.6137695, -5250.5219727, 5720.0830078
2: -1207.4289551, 4676.9970703, -1052.3231201, 4055.3093262, -5262.7382812, 5729.3198242
3: -1901.7006836, 4865.4750977, -1654.4739990, 4221.6499023, -6123.3500977, 6519.9487305
4: -1904.5893555, 4656.3349609, -1659.6434326, 4041.3674316, -5945.9560547, 6315.9785156

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_A2_B1_A1

### Relational analysis result of IS_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9767878, upper bound: 5471.9782402
time: 1.16 seconds

## Relational analysis of IS_A2_A2_B1_A2

### Relational analysis result of IS_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783529, upper bound: 5471.9783529
time: 1.00 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -943.1829224, 4128.3657227, -900.6458740, 3940.8784180, -4884.0615234, 5029.0117188
1: -1187.9083252, 4685.2797852, -1134.3397217, 4471.8212891, -5659.7294922, 5819.6191406
2: -1207.4289551, 4676.9970703, -1153.1384277, 4464.4355469, -5671.8642578, 5830.1352539
3: -1901.7006836, 4865.4750977, -1816.4885254, 4644.1738281, -6545.8745117, 6681.9638672
4: -1904.5893555, 4656.3349609, -1818.9307861, 4446.5278320, -6351.1166992, 6475.2656250

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9767878, upper bound: 5471.9782402
time: 1.08 seconds

## Relational analysis of IS_A2_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783529, upper bound: 5471.9783529
time: 1.00 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.11 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 3, lower bound: -5471.9781246, upper bound: 5471.9792262
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 3, lower bound: -5471.9783662, upper bound: 5471.9792258
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 3, lower bound: -5471.9781246, upper bound: 5471.9792262
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 3, lower bound: -5471.9783662, upper bound: 5471.9792258
IS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 3, lower bound: -5471.9763551, upper bound: 5471.9771959
IS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 3, lower bound: -5471.9783658, upper bound: 5471.9784960
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 3, lower bound: -5471.9770828, upper bound: 5471.9767111
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 3, lower bound: -5471.9783658, upper bound: 5471.9784960
IS_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 3, lower bound: -5471.9792149, upper bound: 5471.9782435
IS_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 3, lower bound: -5471.9792145, upper bound: 5471.9784851
IS_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 3, lower bound: -5471.9792149, upper bound: 5471.9782435
IS_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 3, lower bound: -5471.9792145, upper bound: 5471.9784851
IS_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 3, lower bound: -5471.9767878, upper bound: 5471.9782402
IS_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 3, lower bound: -5471.9783529, upper bound: 5471.9783529
IS_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 3, lower bound: -5471.9767878, upper bound: 5471.9782402
IS_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.11
Output dim: 3, lower bound: -5471.9783529, upper bound: 5471.9783529

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -809.3170776, 3522.6235352, -785.1885376, 3423.9638672, -4233.2807617, 4307.8110352
1: -1019.0266113, 3998.4208984, -989.1842041, 3885.9379883, -4904.9638672, 4987.6040039
2: -1036.0567627, 3991.2778320, -1005.5472412, 3879.8583984, -4915.9150391, 4996.8232422
3: -1629.1091309, 4155.2421875, -1581.5344238, 4037.7421875, -5666.8515625, 5736.7763672
4: -1633.4003906, 3978.4162598, -1586.0260010, 3866.6147461, -5500.0146484, 5564.4404297

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9789142, upper bound: 5471.9789962
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9789142, upper bound: 5471.9792258
time: 1.05 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -809.6653442, 3524.1467285, -816.8909912, 3557.3525391, -4367.0175781, 4341.0366211
1: -1019.4607544, 4000.1494141, -1028.2508545, 4037.4504395, -5056.9111328, 5028.4003906
2: -1036.5018311, 3992.9978027, -1045.6796875, 4030.0437012, -5066.5454102, 5038.6777344
3: -1629.8031006, 4157.0307617, -1644.1126709, 4195.4311523, -5825.2343750, 5801.1435547
4: -1634.1029053, 3980.1218262, -1649.2773438, 4016.0964355, -5650.1992188, 5629.3994141

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9791557, upper bound: 5471.9789962
time: 1.04 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9791557, upper bound: 5471.9792258
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -888.0089111, 3884.5688477, -785.1885376, 3423.9638672, -4311.9726562, 4669.7558594
1: -1118.6939697, 4408.3354492, -989.1842041, 3885.9379883, -5004.6298828, 5397.5190430
2: -1137.0126953, 4401.0244141, -1005.5472412, 3879.8583984, -5016.8701172, 5406.5708008
3: -1791.3597412, 4578.4946289, -1581.5344238, 4037.7421875, -5829.1020508, 6160.0292969
4: -1792.9293213, 4384.2397461, -1586.0260010, 3866.6147461, -5659.5434570, 5970.2656250

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9764226, upper bound: 5471.9790137
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781073, upper bound: 5471.9791041
time: 1.15 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -888.3382568, 3885.9843750, -816.8909912, 3557.3525391, -4445.6904297, 4702.8750000
1: -1119.1064453, 4409.9409180, -1028.2508545, 4037.4504395, -5156.5566406, 5438.1914062
2: -1137.4340820, 4402.6269531, -1045.6796875, 4030.0437012, -5167.4775391, 5448.3066406
3: -1792.0124512, 4580.1577148, -1644.1126709, 4195.4311523, -5987.4433594, 6224.2700195
4: -1793.5903320, 4385.8295898, -1649.2773438, 4016.0964355, -5809.6850586, 6035.1069336

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9765742, upper bound: 5471.9790133
time: 1.04 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782580, upper bound: 5471.9791037
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -772.6435547, 3367.8562012, -900.4553833, 3940.0771484, -4712.7197266, 4268.3115234
1: -973.6626587, 3822.6748047, -1134.1015625, 4470.9135742, -5444.5761719, 4956.7763672
2: -989.5352173, 3816.7854004, -1152.8931885, 4463.5244141, -5453.0595703, 4969.6782227
3: -1556.5802002, 3972.3762207, -1816.1176758, 4643.2324219, -6199.8125000, 5788.4941406
4: -1560.1916504, 3804.6279297, -1818.5528564, 4445.6225586, -6005.8144531, 5623.1787109

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_A1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9789112, upper bound: 5471.9766781
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9790226, upper bound: 5471.9782433
time: 1.01 seconds

## BFS IS instance: IS_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -804.3394775, 3501.3076172, -900.7853394, 3941.4938965, -4745.8330078, 4402.0927734
1: -1012.7216187, 3974.2565918, -1134.5142822, 4472.5200195, -5485.2416992, 5108.7700195
2: -1029.6697998, 3967.0075684, -1153.3151855, 4465.1274414, -5494.7973633, 5120.3227539
3: -1619.1402588, 4130.0576172, -1816.7711182, 4644.8984375, -6264.0385742, 5946.8286133
4: -1623.4382324, 3954.1323242, -1819.2148438, 4447.2148438, -6070.6518555, 5773.3466797

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_A2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9789109, upper bound: 5471.9768295
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9790222, upper bound: 5471.9783946
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -888.0089111, 3884.5688477, -849.9623413, 3720.8525391, -4608.8608398, 4734.5302734
1: -1118.6939697, 4408.3354492, -1071.1772461, 4222.1093750, -5340.8027344, 5479.5122070
2: -1137.0126953, 4401.0244141, -1088.5864258, 4215.8339844, -5352.8466797, 5489.6098633
3: -1791.3597412, 4578.4946289, -1715.1385498, 4385.0781250, -6176.4379883, 6293.6328125
4: -1792.9293213, 4384.2397461, -1717.7025146, 4198.8320312, -5991.7602539, 6101.9423828

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9753986, upper bound: 5471.9764987
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9770824, upper bound: 5471.9765891
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -888.3382568, 3885.9843750, -893.6677856, 3909.9631348, -4798.3012695, 4779.6518555
1: -1119.1064453, 4409.9409180, -1125.5454102, 4436.6674805, -5555.7739258, 5535.4858398
2: -1137.4340820, 4402.6269531, -1144.2006836, 4429.4516602, -5566.8852539, 5546.8271484
3: -1792.0124512, 4580.1577148, -1802.3121338, 4607.6967773, -6399.7089844, 6382.4697266
4: -1793.5903320, 4385.8295898, -1804.7126465, 4411.7182617, -6205.3085938, 6190.5419922

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9765489, upper bound: 5471.9782835
time: 1.15 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9770824, upper bound: 5471.9783739
time: 1.23 seconds

## BFS IS instance: IS_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -825.8685913, 3604.0231934, -821.7229614, 3578.0397949, -4403.9077148, 4425.7460938
1: -1040.4620361, 4090.8273926, -1034.3731689, 4060.9003906, -5101.3623047, 5125.1997070
2: -1057.3305664, 4084.0214844, -1051.8824463, 4053.6049805, -5110.9350586, 5135.9028320
3: -1663.1308594, 4250.3701172, -1653.7860107, 4219.8769531, -5883.0053711, 5904.1552734
4: -1667.7132568, 4067.8308105, -1658.9468994, 4039.6770020, -5707.3901367, 5726.7778320

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 44

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_A1_B1_A1_B1

### Relational analysis result of IS_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9789733, upper bound: 5471.9789734
time: 1.57 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2

### Relational analysis result of IS_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9789733, upper bound: 5471.9789734
time: 1.20 seconds

## BFS IS instance: IS_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -857.7193604, 3737.8354492, -822.0680542, 3579.5488281, -4437.2680664, 4559.9033203
1: -1079.6697998, 4242.9511719, -1034.8034668, 4062.6137695, -5142.2822266, 5277.7543945
2: -1097.7081299, 4234.8027344, -1052.3231201, 4055.3093262, -5153.0170898, 5287.1259766
3: -1725.7722168, 4408.4545898, -1654.4739990, 4221.6499023, -5947.4213867, 6062.9282227
4: -1731.3269043, 4217.8354492, -1659.6434326, 4041.3674316, -5772.6938477, 5877.4785156

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_A1_B1_A2_B1

### Relational analysis result of IS_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9789733, upper bound: 5471.9792149
time: 1.12 seconds

## Relational analysis of IS_A2_A1_B1_A2_B2

### Relational analysis result of IS_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9789733, upper bound: 5471.9792149
time: 1.06 seconds

## BFS IS instance: IS_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -825.8685913, 3604.0231934, -900.3157959, 3939.4611816, -4765.3295898, 4504.3388672
1: -1040.4620361, 4090.8273926, -1133.9273682, 4470.2143555, -5510.6762695, 5224.7534180
2: -1057.3305664, 4084.0214844, -1152.7163086, 4462.8310547, -5520.1606445, 5236.7368164
3: -1663.1308594, 4250.3701172, -1815.8350830, 4642.5083008, -6305.6376953, 6066.2050781
4: -1667.7132568, 4067.8308105, -1818.2690430, 4444.9365234, -6112.6499023, 5886.0991211

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_A1_B2_A1_B1

### Relational analysis result of IS_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9789705, upper bound: 5471.9766611
time: 0.98 seconds

## Relational analysis of IS_A2_A1_B2_A1_B2

### Relational analysis result of IS_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9790819, upper bound: 5471.9782262
time: 1.13 seconds

## BFS IS instance: IS_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -857.7193604, 3737.8354492, -900.6458740, 3940.8784180, -4798.5976562, 4638.4814453
1: -1079.6697998, 4242.9511719, -1134.3397217, 4471.8212891, -5551.4912109, 5377.2905273
2: -1097.7081299, 4234.8027344, -1153.1384277, 4464.4355469, -5562.1435547, 5387.9409180
3: -1725.7722168, 4408.4545898, -1816.4885254, 4644.1738281, -6369.9462891, 6224.9433594
4: -1731.3269043, 4217.8354492, -1818.9307861, 4446.5278320, -6177.8544922, 6036.7661133

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_A1_B2_A2_B1

### Relational analysis result of IS_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9789701, upper bound: 5471.9768118
time: 1.00 seconds

## Relational analysis of IS_A2_A1_B2_A2_B2

### Relational analysis result of IS_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9790815, upper bound: 5471.9783769
time: 0.96 seconds

## BFS IS instance: IS_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -943.6794434, 4133.3291016, -820.0695801, 3570.7023926, -4514.3808594, 4953.3984375
1: -1188.1201172, 4691.8979492, -1032.2852783, 4052.5517578, -5240.6718750, 5724.1826172
2: -1207.8543701, 4681.4077148, -1049.7769775, 4045.3291016, -5253.1835938, 5731.1845703
3: -1902.2111816, 4871.1386719, -1650.3803711, 4211.1982422, -6113.4091797, 6521.5190430
4: -1905.7425537, 4659.7895508, -1655.5686035, 4031.4248047, -5937.1674805, 6315.3579102

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_A2_B1_A1_A1

### Relational analysis result of IS_A2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9728106, upper bound: 5471.9769581
time: 0.98 seconds

## Relational analysis of IS_A2_A2_B1_A1_A2

### Relational analysis result of IS_A2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9768118, upper bound: 5471.9789701
time: 0.99 seconds

## BFS IS instance: IS_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -938.3684082, 4107.0532227, -822.0680542, 3579.5488281, -4517.9169922, 4929.1210938
1: -1181.8682861, 4661.0883789, -1034.8034668, 4062.6137695, -5244.4819336, 5695.8916016
2: -1201.3096924, 4652.9384766, -1052.3231201, 4055.3093262, -5256.6191406, 5705.2612305
3: -1892.0146484, 4840.4399414, -1654.4739990, 4221.6499023, -6113.6645508, 6494.9130859
4: -1894.8724365, 4632.5317383, -1659.6434326, 4041.3674316, -5936.2387695, 6292.1748047

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_A2_B1_A2_A1

### Relational analysis result of IS_A2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9765886, upper bound: 5471.9779311
time: 1.03 seconds

## Relational analysis of IS_A2_A2_B1_A2_A2

### Relational analysis result of IS_A2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783769, upper bound: 5471.9790814
time: 0.95 seconds

## BFS IS instance: IS_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -943.6794434, 4133.3291016, -898.4242554, 3930.9318848, -4874.6103516, 5031.7534180
1: -1188.1201172, 4691.8979492, -1131.5408936, 4460.5200195, -5648.6396484, 5823.4384766
2: -1207.8543701, 4681.4077148, -1150.3101807, 4453.2119141, -5661.0664062, 5831.7167969
3: -1902.2111816, 4871.1386719, -1811.9221191, 4632.4453125, -6534.6557617, 6683.0605469
4: -1905.7425537, 4659.7895508, -1814.4030762, 4435.3608398, -6341.1035156, 6474.1918945

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_A2_B2_A1_A1

### Relational analysis result of IS_A2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9727944, upper bound: 5471.9762283
time: 1.30 seconds

## Relational analysis of IS_A2_A2_B2_A1_A2

### Relational analysis result of IS_A2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9767865, upper bound: 5471.9782402
time: 1.17 seconds

## BFS IS instance: IS_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -938.3684082, 4107.0532227, -900.6458740, 3940.8784180, -4879.2470703, 5007.6992188
1: -1181.8682861, 4661.0883789, -1134.3397217, 4471.8212891, -5653.6894531, 5795.4282227
2: -1201.3096924, 4652.9384766, -1153.1384277, 4464.4355469, -5665.7451172, 5806.0771484
3: -1892.0146484, 4840.4399414, -1816.4885254, 4644.1738281, -6536.1884766, 6656.9282227
4: -1894.8724365, 4632.5317383, -1818.9307861, 4446.5278320, -6341.3989258, 6451.4624023

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_A2_B2_A2_A1

### Relational analysis result of IS_A2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9765664, upper bound: 5471.9772013
time: 0.95 seconds

## Relational analysis of IS_A2_A2_B2_A2_A2

### Relational analysis result of IS_A2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783516, upper bound: 5471.9783516
time: 0.98 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.99 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.99
Output dim: 3, lower bound: -5471.9789142, upper bound: 5471.9789962
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.99
Output dim: 3, lower bound: -5471.9789142, upper bound: 5471.9792258
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.99
Output dim: 3, lower bound: -5471.9791557, upper bound: 5471.9789962
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.99
Output dim: 3, lower bound: -5471.9791557, upper bound: 5471.9792258
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.99
Output dim: 3, lower bound: -5471.9764226, upper bound: 5471.9790137
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.99
Output dim: 3, lower bound: -5471.9781073, upper bound: 5471.9791041
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.99
Output dim: 3, lower bound: -5471.9765742, upper bound: 5471.9790133
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.99
Output dim: 3, lower bound: -5471.9782580, upper bound: 5471.9791037
IS_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.99
Output dim: 3, lower bound: -5471.9789112, upper bound: 5471.9766781
IS_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.99
Output dim: 3, lower bound: -5471.9790226, upper bound: 5471.9782433
IS_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.99
Output dim: 3, lower bound: -5471.9789109, upper bound: 5471.9768295
IS_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.99
Output dim: 3, lower bound: -5471.9790222, upper bound: 5471.9783946
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.99
Output dim: 3, lower bound: -5471.9753986, upper bound: 5471.9764987
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.99
Output dim: 3, lower bound: -5471.9770824, upper bound: 5471.9765891
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.99
Output dim: 3, lower bound: -5471.9765489, upper bound: 5471.9782835
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.99
Output dim: 3, lower bound: -5471.9770824, upper bound: 5471.9783739
IS_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.99
Output dim: 3, lower bound: -5471.9789733, upper bound: 5471.9789734
IS_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.99
Output dim: 3, lower bound: -5471.9789733, upper bound: 5471.9789734
IS_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.99
Output dim: 3, lower bound: -5471.9789733, upper bound: 5471.9792149
IS_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.99
Output dim: 3, lower bound: -5471.9789733, upper bound: 5471.9792149
IS_A2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.99
Output dim: 3, lower bound: -5471.9789705, upper bound: 5471.9766611
IS_A2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.99
Output dim: 3, lower bound: -5471.9790819, upper bound: 5471.9782262
IS_A2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.99
Output dim: 3, lower bound: -5471.9789701, upper bound: 5471.9768118
IS_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.99
Output dim: 3, lower bound: -5471.9790815, upper bound: 5471.9783769
IS_A2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.99
Output dim: 3, lower bound: -5471.9728106, upper bound: 5471.9769581
IS_A2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.99
Output dim: 3, lower bound: -5471.9768118, upper bound: 5471.9789701
IS_A2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.99
Output dim: 3, lower bound: -5471.9765886, upper bound: 5471.9779311
IS_A2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.99
Output dim: 3, lower bound: -5471.9783769, upper bound: 5471.9790814
IS_A2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.99
Output dim: 3, lower bound: -5471.9727944, upper bound: 5471.9762283
IS_A2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.99
Output dim: 3, lower bound: -5471.9767865, upper bound: 5471.9782402
IS_A2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.99
Output dim: 3, lower bound: -5471.9765664, upper bound: 5471.9772013
IS_A2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.99
Output dim: 3, lower bound: -5471.9783516, upper bound: 5471.9783516

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -772.6435547, 3367.8562012, -785.1885376, 3423.9638672, -4196.6069336, 4153.0444336
1: -973.6626587, 3822.6748047, -989.1842041, 3885.9379883, -4859.6000977, 4811.8588867
2: -989.5352173, 3816.7854004, -1005.5472412, 3879.8583984, -4869.3930664, 4822.3320312
3: -1556.5802002, 3972.3762207, -1581.5344238, 4037.7421875, -5594.3222656, 5553.9106445
4: -1560.1916504, 3804.6279297, -1586.0260010, 3866.6147461, -5426.8066406, 5390.6528320

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780552, upper bound: 5471.9776502
time: 1.05 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9774441, upper bound: 5471.9775810
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -804.3394775, 3501.3076172, -785.1885376, 3423.9638672, -4228.3032227, 4286.4956055
1: -1012.7216187, 3974.2565918, -989.1842041, 3885.9379883, -4898.6591797, 4963.4394531
2: -1029.6697998, 3967.0075684, -1005.5472412, 3879.8583984, -4909.5283203, 4972.5541992
3: -1619.1402588, 4130.0576172, -1581.5344238, 4037.7421875, -5656.8823242, 5711.5917969
4: -1623.4382324, 3954.1323242, -1586.0260010, 3866.6147461, -5490.0527344, 5540.1582031

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780552, upper bound: 5471.9778745
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9774441, upper bound: 5471.9777519
time: 1.21 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -772.6435547, 3367.8562012, -816.8909912, 3557.3525391, -4329.9951172, 4184.7470703
1: -973.6626587, 3822.6748047, -1028.2508545, 4037.4504395, -5011.1132812, 4850.9257812
2: -989.5352173, 3816.7854004, -1045.6796875, 4030.0437012, -5019.5791016, 4862.4648438
3: -1556.5802002, 3972.3762207, -1644.1126709, 4195.4311523, -5752.0112305, 5616.4887695
4: -1560.1916504, 3804.6279297, -1649.2773438, 4016.0964355, -5576.2875977, 5453.9052734

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9775473, upper bound: 5471.9782471
time: 1.07 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9774441, upper bound: 5471.9775894
time: 1.17 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -804.3394775, 3501.3076172, -816.8909912, 3557.3525391, -4361.6914062, 4318.1987305
1: -1012.7216187, 3974.2565918, -1028.2508545, 4037.4504395, -5050.1718750, 5002.5068359
2: -1029.6697998, 3967.0075684, -1045.6796875, 4030.0437012, -5059.7133789, 5012.6875000
3: -1619.1402588, 4130.0576172, -1644.1126709, 4195.4311523, -5814.5712891, 5774.1689453
4: -1623.4382324, 3954.1323242, -1649.2773438, 4016.0964355, -5639.5322266, 5603.4096680

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780552, upper bound: 5471.9778825
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9774307, upper bound: 5471.9777604
time: 1.06 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -888.5172119, 3889.9057617, -783.2211914, 3415.2158203, -4303.7329102, 4673.1259766
1: -1118.9589844, 4415.5283203, -986.7105713, 3875.9697266, -4994.9287109, 5402.2387695
2: -1137.4306641, 4406.0126953, -1003.0382080, 3869.9887695, -5007.4194336, 5409.0498047
3: -1792.0482178, 4584.5644531, -1577.5126953, 4027.4045410, -5819.4526367, 6162.0771484
4: -1794.0584717, 4388.4980469, -1582.0108643, 3856.7988281, -5650.8574219, 5970.5078125

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B1_A1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9718930, upper bound: 5471.9769494
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9718930, upper bound: 5471.9790137
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -882.9274292, 3862.1269531, -785.1885376, 3423.9638672, -4306.8911133, 4647.3139648
1: -1112.3194580, 4382.8598633, -989.1842041, 3885.9379883, -4998.2553711, 5372.0429688
2: -1130.5550537, 4375.6972656, -1005.5472412, 3879.8583984, -5010.4130859, 5381.2441406
3: -1781.1387939, 4552.1186523, -1581.5344238, 4037.7421875, -5818.8808594, 6133.6528320
4: -1782.6708984, 4359.1699219, -1586.0260010, 3866.6147461, -5649.2856445, 5945.1958008

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9760962, upper bound: 5471.9779262
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9760962, upper bound: 5471.9779262
time: 1.14 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -888.8275146, 3891.2319336, -814.9169922, 3548.6230469, -4437.4506836, 4706.1479492
1: -1119.3469238, 4417.0312500, -1025.7645264, 4027.5227051, -5146.8696289, 5442.7958984
2: -1137.8276367, 4407.5161133, -1043.1665039, 4020.1958008, -5158.0229492, 5450.6816406
3: -1792.6622314, 4586.1245117, -1640.0700684, 4185.1201172, -5977.7822266, 6226.1943359
4: -1794.6807861, 4389.9921875, -1645.2575684, 4006.2822266, -5800.9614258, 6035.2485352

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B2_A1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9718930, upper bound: 5471.9769490
time: 1.15 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9718930, upper bound: 5471.9769490
time: 1.49 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -883.2570190, 3863.5400391, -816.8909912, 3557.3525391, -4440.6093750, 4680.4301758
1: -1112.7313232, 4384.4619141, -1028.2508545, 4037.4504395, -5150.1811523, 5412.7128906
2: -1130.9761963, 4377.2978516, -1045.6796875, 4030.0437012, -5161.0200195, 5422.9775391
3: -1781.7916260, 4553.7802734, -1644.1126709, 4195.4311523, -5977.2226562, 6197.8920898
4: -1783.3317871, 4360.7568359, -1649.2773438, 4016.0964355, -5799.4257812, 6010.0341797

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B2_A2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9760962, upper bound: 5471.9779257
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9760962, upper bound: 5471.9791037
time: 1.04 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -770.6763306, 3359.1083984, -900.9668579, 3945.3735352, -4716.0498047, 4260.0751953
1: -971.1892090, 3812.7119141, -1134.3739014, 4478.0888672, -5449.2783203, 4947.0859375
2: -987.0273438, 3806.9226074, -1153.3175049, 4468.4814453, -5455.5083008, 4960.2402344
3: -1552.5593262, 3962.0441895, -1816.8322754, 4649.2939453, -6201.8535156, 5778.8764648
4: -1556.1773682, 3794.8190918, -1819.6914062, 4449.8916016, -6006.0688477, 5614.5102539

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9768993, upper bound: 5471.9726520
time: 1.34 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9768993, upper bound: 5471.9766781
time: 0.94 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -772.6435547, 3367.8562012, -895.3926392, 3917.7338867, -4690.3769531, 4263.2490234
1: -973.6626587, 3822.6748047, -1127.7513428, 4445.5507812, -5419.2133789, 4950.4262695
2: -989.5352173, 3816.7854004, -1146.4605713, 4438.3120117, -5427.8466797, 4963.2456055
3: -1556.5802002, 3972.3762207, -1805.9393311, 4616.9711914, -6173.5512695, 5778.3154297
4: -1560.1916504, 3804.6279297, -1808.3364258, 4420.6674805, -5980.8583984, 5612.9628906

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9778723, upper bound: 5471.9764578
time: 1.05 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9778723, upper bound: 5471.9782433
time: 2.64 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -802.3634644, 3492.5703125, -901.2770996, 3946.7004395, -4749.0634766, 4393.8476562
1: -1010.2330933, 3964.3205566, -1134.7617188, 4479.5932617, -5489.8261719, 5099.0820312
2: -1027.1541748, 3957.1511230, -1153.7145996, 4469.9858398, -5497.1391602, 5110.8657227
3: -1615.0947266, 4119.7353516, -1817.4465332, 4650.8554688, -6265.9501953, 5937.1816406
4: -1619.4149170, 3944.3100586, -1820.3132324, 4451.3852539, -6070.7993164, 5764.6230469

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_A2_B1_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9768989, upper bound: 5471.9728267
time: 1.27 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_B2

### Relational analysis result of IS_A1_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9768989, upper bound: 5471.9768295
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -804.3394775, 3501.3076172, -895.7224731, 3919.1472168, -4723.4868164, 4397.0302734
1: -1012.7216187, 3974.2565918, -1128.1638184, 4447.1528320, -5459.8745117, 5102.4204102
2: -1029.6697998, 3967.0075684, -1146.8822021, 4439.9125977, -5469.5825195, 5113.8886719
3: -1619.1402588, 4130.0576172, -1806.5917969, 4618.6337891, -6237.7734375, 5936.6489258
4: -1623.4382324, 3954.1323242, -1808.9970703, 4422.2539062, -6045.6914062, 5763.1289062

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_A2_B2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9778719, upper bound: 5471.9766090
time: 1.07 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9778719, upper bound: 5471.9783946
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -888.5172119, 3889.9057617, -847.8782349, 3711.4760742, -4599.9931641, 4737.7836914
1: -1118.9589844, 4415.5283203, -1068.5543213, 4211.4501953, -5330.4091797, 5484.0820312
2: -1137.4306641, 4406.0126953, -1085.9305420, 4205.2583008, -5342.6884766, 5491.9423828
3: -1792.0482178, 4584.5644531, -1710.8579102, 4374.0307617, -6166.0781250, 6295.4223633
4: -1794.0584717, 4388.4980469, -1713.4458008, 4188.3178711, -5982.3764648, 6101.9438477

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B1_A1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9709035, upper bound: 5471.9744344
time: 1.27 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9709035, upper bound: 5471.9764987
time: 1.36 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -882.9274292, 3862.1269531, -849.9623413, 3720.8525391, -4603.7792969, 4712.0883789
1: -1112.3194580, 4382.8598633, -1071.1772461, 4222.1093750, -5334.4282227, 5454.0361328
2: -1130.5550537, 4375.6972656, -1088.5864258, 4215.8339844, -5346.3891602, 5464.2827148
3: -1781.1387939, 4552.1186523, -1715.1385498, 4385.0781250, -6166.2163086, 6267.2568359
4: -1782.6708984, 4359.1699219, -1717.7025146, 4198.8320312, -5981.5029297, 6076.8720703

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B1_A2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9750717, upper bound: 5471.9754112
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9750717, upper bound: 5471.9765891
time: 1.14 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -888.8275146, 3891.2319336, -891.4517822, 3900.0568848, -4788.8837891, 4782.6831055
1: -1119.3469238, 4417.0312500, -1122.7525635, 4425.4111328, -5544.7578125, 5539.7836914
2: -1137.8276367, 4407.5161133, -1141.3802490, 4418.2700195, -5556.0976562, 5548.8940430
3: -1792.6622314, 4586.1245117, -1797.7586670, 4596.0146484, -6388.6767578, 6383.8833008
4: -1794.6807861, 4389.9921875, -1800.1989746, 4400.5913086, -6195.2714844, 6190.1889648

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B2_A1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9709035, upper bound: 5471.9762192
time: 1.05 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9709035, upper bound: 5471.9782835
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -883.2570190, 3863.5400391, -893.6677856, 3909.9631348, -4793.2202148, 4757.2070312
1: -1112.7313232, 4384.4619141, -1125.5454102, 4436.6674805, -5549.3984375, 5510.0073242
2: -1130.9761963, 4377.2978516, -1144.2006836, 4429.4516602, -5560.4277344, 5521.4975586
3: -1781.7916260, 4553.7802734, -1802.3121338, 4607.6967773, -6389.4882812, 6356.0922852
4: -1783.3317871, 4360.7568359, -1804.7126465, 4411.7182617, -6195.0498047, 6165.4692383

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9750717, upper bound: 5471.9771959
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9750717, upper bound: 5471.9783739
time: 0.93 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -825.8685913, 3604.0231934, -785.0410767, 3423.3166504, -4249.1850586, 4389.0644531
1: -1040.4620361, 4090.8273926, -988.9992065, 3885.2014160, -4925.6635742, 5079.8261719
2: -1057.3305664, 4084.0214844, -1005.3588867, 3879.1298828, -4936.4589844, 5089.3779297
3: -1663.1308594, 4250.3701172, -1581.2360840, 4036.9787598, -5700.1088867, 5831.6064453
4: -1667.7132568, 4067.8308105, -1585.7248535, 3865.8928223, -5533.6059570, 5653.5551758

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781903, upper bound: 5471.9776274
time: 1.32 seconds

## Relational analysis of IS_A2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9775242, upper bound: 5471.9775242
time: 1.14 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -825.8685913, 3604.0231934, -816.7496338, 3556.7277832, -4382.5961914, 4420.7729492
1: -1040.4620361, 4090.8273926, -1028.0739746, 4036.7412109, -5077.2031250, 5118.9013672
2: -1057.3305664, 4084.0214844, -1045.4992676, 4029.3427734, -5086.6728516, 5129.5200195
3: -1663.1308594, 4250.3701172, -1643.8266602, 4194.6962891, -5857.8256836, 5894.1962891
4: -1667.7132568, 4067.8308105, -1648.9881592, 4015.4025879, -5683.1152344, 5716.8188477

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_B1_A1_B2_B1

### Relational analysis result of IS_A2_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9786486, upper bound: 5471.9780956
time: 1.18 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2_B2

### Relational analysis result of IS_A2_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9789737, upper bound: 5471.9789734
time: 1.16 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -857.7193604, 3737.8354492, -785.0410767, 3423.3166504, -4281.0361328, 4522.8764648
1: -1079.6697998, 4242.9511719, -988.9992065, 3885.2014160, -4964.8696289, 5231.9501953
2: -1097.7081299, 4234.8027344, -1005.3588867, 3879.1298828, -4976.8364258, 5240.1606445
3: -1725.7722168, 4408.4545898, -1581.2360840, 4036.9787598, -5762.7509766, 5989.6904297
4: -1731.3269043, 4217.8354492, -1585.7248535, 3865.8928223, -5597.2197266, 5803.5595703

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781903, upper bound: 5471.9778664
time: 1.03 seconds

## Relational analysis of IS_A2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9775326, upper bound: 5471.9777259
time: 1.47 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -857.7193604, 3737.8354492, -816.7496338, 3556.7277832, -4414.4472656, 4554.5849609
1: -1079.6697998, 4242.9511719, -1028.0739746, 4036.7412109, -5116.4096680, 5271.0253906
2: -1097.7081299, 4234.8027344, -1045.4992676, 4029.3427734, -5127.0498047, 5280.3017578
3: -1725.7722168, 4408.4545898, -1643.8266602, 4194.6962891, -5920.4682617, 6052.2812500
4: -1731.3269043, 4217.8354492, -1648.9881592, 4015.4025879, -5746.7285156, 5866.8237305

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781903, upper bound: 5471.9778744
time: 0.95 seconds

## Relational analysis of IS_A2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9775326, upper bound: 5471.9777343
time: 1.16 seconds

## BFS IS instance: IS_A2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -823.9197998, 3595.3574219, -900.8369141, 3944.7878418, -4768.7075195, 4496.1943359
1: -1038.0114746, 4080.9526367, -1134.2110596, 4477.4238281, -5515.4355469, 5215.1635742
2: -1054.8452148, 4074.2514648, -1153.1518555, 4467.8251953, -5522.6704102, 5227.4018555
3: -1659.1478271, 4240.1298828, -1816.5671387, 4648.6064453, -6307.7543945, 6056.6972656
4: -1663.7410889, 4058.1076660, -1819.4244385, 4449.2416992, -6112.9829102, 5877.5322266

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9787738, upper bound: 5471.9764112
time: 1.05 seconds

## Relational analysis of IS_A2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9789669, upper bound: 5471.9765986
time: 0.96 seconds

## BFS IS instance: IS_A2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -825.8685913, 3604.0231934, -895.2587280, 3917.1433105, -4743.0112305, 4499.2817383
1: -1040.4620361, 4090.8273926, -1127.5842285, 4444.8789062, -5485.3408203, 5218.4116211
2: -1057.3305664, 4084.0214844, -1146.2910156, 4437.6464844, -5494.9765625, 5230.3110352
3: -1663.1308594, 4250.3701172, -1805.6678467, 4616.2768555, -6279.4062500, 6056.0380859
4: -1667.7132568, 4067.8308105, -1808.0642090, 4420.0083008, -6087.7216797, 5875.8940430

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9788888, upper bound: 5471.9780123
time: 1.06 seconds

## Relational analysis of IS_A2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9790819, upper bound: 5471.9781991
time: 1.04 seconds

## BFS IS instance: IS_A2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -855.7315063, 3728.9936523, -901.1470947, 3946.1147461, -4801.8461914, 4630.1406250
1: -1077.1621094, 4232.8891602, -1134.5987549, 4478.9287109, -5556.0908203, 5367.4877930
2: -1095.1730957, 4224.8300781, -1153.5485840, 4469.3295898, -5564.5009766, 5378.3784180
3: -1721.6938477, 4398.0200195, -1817.1812744, 4650.1674805, -6371.8613281, 6215.2011719
4: -1727.2734375, 4207.9013672, -1820.0466309, 4450.7353516, -6178.0083008, 6027.9482422

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_A1_B2_A2_B1_B1

### Relational analysis result of IS_A2_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9769581, upper bound: 5471.9728106
time: 0.98 seconds

## Relational analysis of IS_A2_A1_B2_A2_B1_B2

### Relational analysis result of IS_A2_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9769581, upper bound: 5471.9768099
time: 0.91 seconds

## BFS IS instance: IS_A2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -857.7193604, 3737.8354492, -895.5885010, 3918.5561523, -4776.2749023, 4633.4238281
1: -1079.6697998, 4242.9511719, -1127.9964600, 4446.4819336, -5526.1513672, 5370.9472656
2: -1097.7081299, 4234.8027344, -1146.7124023, 4439.2470703, -5536.9541016, 5381.5146484
3: -1725.7722168, 4408.4545898, -1806.3208008, 4617.9389648, -6343.7109375, 6214.7753906
4: -1731.3269043, 4217.8354492, -1808.7253418, 4421.5961914, -6152.9223633, 6026.5600586

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_A1_B2_A2_B2_B1

### Relational analysis result of IS_A2_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9779311, upper bound: 5471.9765886
time: 1.37 seconds

## Relational analysis of IS_A2_A1_B2_A2_B2_B2

### Relational analysis result of IS_A2_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9779311, upper bound: 5471.9783769
time: 1.03 seconds

## BFS IS instance: IS_A2_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -891.4138184, 3905.6472168, -819.7196045, 3569.1713867, -4460.5844727, 4725.3666992
1: -1123.1735840, 4432.9287109, -1031.8487549, 4050.8127441, -5173.9863281, 5464.7768555
2: -1141.3878174, 4424.8149414, -1049.3294678, 4043.5998535, -5184.9877930, 5474.1440430
3: -1797.9030762, 4602.9570312, -1649.6823730, 4209.3969727, -6007.2998047, 6252.6391602
4: -1801.0560303, 4404.8403320, -1654.8619385, 4029.7102051, -5830.7661133, 6059.7011719

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_A2_B1_A1_A1_B1

### Relational analysis result of IS_A2_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9726361, upper bound: 5471.9769581
time: 1.01 seconds

## Relational analysis of IS_A2_A2_B1_A1_A1_B2

### Relational analysis result of IS_A2_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9726361, upper bound: 5471.9769582
time: 1.19 seconds

## BFS IS instance: IS_A2_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -935.8339233, 4098.3437500, -820.0695801, 3570.7023926, -4506.5361328, 4918.4130859
1: -1178.2315674, 4652.1391602, -1032.2852783, 4052.5517578, -5230.7832031, 5684.4238281
2: -1197.8010254, 4641.8608398, -1049.7769775, 4045.3291016, -5243.1298828, 5691.6367188
3: -1886.2252197, 4829.9155273, -1650.3803711, 4211.1982422, -6097.4228516, 6480.2958984
4: -1889.7298584, 4620.4165039, -1655.5686035, 4031.4248047, -5921.1547852, 6275.9853516

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_A2_B1_A1_A2_B1

### Relational analysis result of IS_A2_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9766611, upper bound: 5471.9789701
time: 1.03 seconds

## Relational analysis of IS_A2_A2_B1_A1_A2_B2

### Relational analysis result of IS_A2_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9766611, upper bound: 5471.9789701
time: 0.95 seconds

## BFS IS instance: IS_A2_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -886.9016113, 3883.7211914, -821.7229614, 3578.0397949, -4464.9399414, 4705.4443359
1: -1117.7502441, 4407.6523438, -1034.3731689, 4060.9003906, -5178.6503906, 5442.0249023
2: -1135.7454834, 4400.5468750, -1051.8824463, 4053.6049805, -5189.3491211, 5452.4287109
3: -1789.1848145, 4577.5278320, -1653.7860107, 4219.8769531, -6009.0615234, 6231.3129883
4: -1791.9608154, 4381.1083984, -1658.9468994, 4039.6770020, -5831.6367188, 6040.0551758

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_A2_B1_A2_A1_B1

### Relational analysis result of IS_A2_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9764410, upper bound: 5471.9779311
time: 1.02 seconds

## Relational analysis of IS_A2_A2_B1_A2_A1_B2

### Relational analysis result of IS_A2_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9764410, upper bound: 5471.9779311
time: 1.01 seconds

## BFS IS instance: IS_A2_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -931.1712036, 4075.0515137, -822.0680542, 3579.5488281, -4510.7202148, 4897.1196289
1: -1172.8034668, 4624.6962891, -1034.8034668, 4062.6137695, -5235.4165039, 5659.4995117
2: -1192.0935059, 4616.7319336, -1052.3231201, 4055.3093262, -5247.4028320, 5669.0541992
3: -1877.3706055, 4802.7070312, -1654.4739990, 4221.6499023, -6099.0190430, 6457.1811523
4: -1880.1920166, 4596.5097656, -1659.6434326, 4041.3674316, -5921.5585938, 6256.1528320

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_A2_B1_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782262, upper bound: 5471.9790814
time: 1.17 seconds

## Relational analysis of IS_A2_A2_B1_A2_A2_B2

### Relational analysis result of IS_A2_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782262, upper bound: 5471.9790815
time: 1.04 seconds

## BFS IS instance: IS_A2_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -891.4138184, 3905.6472168, -898.0955200, 3929.5205078, -4820.9331055, 4803.7426758
1: -1123.1735840, 4432.9287109, -1131.1297607, 4458.9189453, -5582.0927734, 5564.0585938
2: -1141.3878174, 4424.8149414, -1149.8897705, 4451.6123047, -5593.0000000, 5574.7036133
3: -1797.9030762, 4602.9570312, -1811.2711182, 4630.7851562, -6428.6884766, 6414.2275391
4: -1801.0560303, 4404.8403320, -1813.7436523, 4433.7739258, -6234.8295898, 6218.5834961

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_A2_B2_A1_A1_B1

### Relational analysis result of IS_A2_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9716441, upper bound: 5471.9744436
time: 1.12 seconds

## Relational analysis of IS_A2_A2_B2_A1_A1_B2

### Relational analysis result of IS_A2_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9716441, upper bound: 5471.9762283
time: 1.07 seconds

## BFS IS instance: IS_A2_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -935.8339233, 4098.3437500, -898.4242554, 3930.9318848, -4866.7656250, 4996.7680664
1: -1178.2315674, 4652.1391602, -1131.5408936, 4460.5200195, -5638.7514648, 5783.6796875
2: -1197.8010254, 4641.8608398, -1150.3101807, 4453.2119141, -5651.0126953, 5792.1699219
3: -1886.2252197, 4829.9155273, -1811.9221191, 4632.4453125, -6518.6694336, 6641.8378906
4: -1889.7298584, 4620.4165039, -1814.4030762, 4435.3608398, -6325.0908203, 6434.8193359

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_A2_B2_A1_A2_B1

### Relational analysis result of IS_A2_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756362, upper bound: 5471.9764555
time: 0.96 seconds

## Relational analysis of IS_A2_A2_B2_A1_A2_B2

### Relational analysis result of IS_A2_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756362, upper bound: 5471.9782402
time: 1.08 seconds

## BFS IS instance: IS_A2_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -886.9016113, 3883.7211914, -900.3157959, 3939.4611816, -4826.3627930, 4784.0371094
1: -1117.7502441, 4407.6523438, -1133.9273682, 4470.2143555, -5587.9648438, 5541.5791016
2: -1135.7454834, 4400.5468750, -1152.7163086, 4462.8310547, -5598.5747070, 5553.2622070
3: -1789.1848145, 4577.5278320, -1815.8350830, 4642.5083008, -6431.6933594, 6393.3627930
4: -1791.9608154, 4381.1083984, -1818.2690430, 4444.9365234, -6236.8969727, 6199.3774414

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_A2_B2_A2_A1_B1

### Relational analysis result of IS_A2_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754166, upper bound: 5471.9754166
time: 1.15 seconds

## Relational analysis of IS_A2_A2_B2_A2_A1_B2

### Relational analysis result of IS_A2_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9754166, upper bound: 5471.9772013
time: 1.19 seconds

## BFS IS instance: IS_A2_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -931.1712036, 4075.0515137, -900.6458740, 3940.8784180, -4872.0498047, 4975.6972656
1: -1172.8034668, 4624.6962891, -1134.3397217, 4471.8212891, -5644.6245117, 5759.0356445
2: -1192.0935059, 4616.7319336, -1153.1384277, 4464.4355469, -5656.5292969, 5769.8696289
3: -1877.3706055, 4802.7070312, -1816.4885254, 4644.1738281, -6521.5439453, 6619.1953125
4: -1880.1920166, 4596.5097656, -1818.9307861, 4446.5278320, -6326.7197266, 6415.4399414

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_A2_B2_A2_A2_B1

### Relational analysis result of IS_A2_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9772013, upper bound: 5471.9765664
time: 0.99 seconds

## Relational analysis of IS_A2_A2_B2_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9772013, upper bound: 5471.9783516
time: 1.01 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.16 seconds
IS_A1_B1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9780552, upper bound: 5471.9776502
IS_A1_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9774441, upper bound: 5471.9775810
IS_A1_B1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9780552, upper bound: 5471.9778745
IS_A1_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9774441, upper bound: 5471.9777519
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9775473, upper bound: 5471.9782471
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9774441, upper bound: 5471.9775894
IS_A1_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9780552, upper bound: 5471.9778825
IS_A1_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9774307, upper bound: 5471.9777604
IS_A1_B1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9718930, upper bound: 5471.9769494
IS_A1_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9718930, upper bound: 5471.9790137
IS_A1_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9760962, upper bound: 5471.9779262
IS_A1_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9760962, upper bound: 5471.9779262
IS_A1_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9718930, upper bound: 5471.9769490
IS_A1_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9718930, upper bound: 5471.9769490
IS_A1_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9760962, upper bound: 5471.9779257
IS_A1_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9760962, upper bound: 5471.9791037
IS_A1_B2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9768993, upper bound: 5471.9726520
IS_A1_B2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9768993, upper bound: 5471.9766781
IS_A1_B2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9778723, upper bound: 5471.9764578
IS_A1_B2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9778723, upper bound: 5471.9782433
IS_A1_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9768989, upper bound: 5471.9728267
IS_A1_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9768989, upper bound: 5471.9768295
IS_A1_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9778719, upper bound: 5471.9766090
IS_A1_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9778719, upper bound: 5471.9783946
IS_A1_B2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9709035, upper bound: 5471.9744344
IS_A1_B2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9709035, upper bound: 5471.9764987
IS_A1_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9750717, upper bound: 5471.9754112
IS_A1_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9750717, upper bound: 5471.9765891
IS_A1_B2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9709035, upper bound: 5471.9762192
IS_A1_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9709035, upper bound: 5471.9782835
IS_A1_B2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9750717, upper bound: 5471.9771959
IS_A1_B2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9750717, upper bound: 5471.9783739
IS_A2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9781903, upper bound: 5471.9776274
IS_A2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9775242, upper bound: 5471.9775242
IS_A2_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9786486, upper bound: 5471.9780956
IS_A2_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9789737, upper bound: 5471.9789734
IS_A2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9781903, upper bound: 5471.9778664
IS_A2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9775326, upper bound: 5471.9777259
IS_A2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9781903, upper bound: 5471.9778744
IS_A2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9775326, upper bound: 5471.9777343
IS_A2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9787738, upper bound: 5471.9764112
IS_A2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9789669, upper bound: 5471.9765986
IS_A2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9788888, upper bound: 5471.9780123
IS_A2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9790819, upper bound: 5471.9781991
IS_A2_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9769581, upper bound: 5471.9728106
IS_A2_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9769581, upper bound: 5471.9768099
IS_A2_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9779311, upper bound: 5471.9765886
IS_A2_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9779311, upper bound: 5471.9783769
IS_A2_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9726361, upper bound: 5471.9769581
IS_A2_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9726361, upper bound: 5471.9769582
IS_A2_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9766611, upper bound: 5471.9789701
IS_A2_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9766611, upper bound: 5471.9789701
IS_A2_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9764410, upper bound: 5471.9779311
IS_A2_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9764410, upper bound: 5471.9779311
IS_A2_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9782262, upper bound: 5471.9790814
IS_A2_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9782262, upper bound: 5471.9790815
IS_A2_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9716441, upper bound: 5471.9744436
IS_A2_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9716441, upper bound: 5471.9762283
IS_A2_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9756362, upper bound: 5471.9764555
IS_A2_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9756362, upper bound: 5471.9782402
IS_A2_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9754166, upper bound: 5471.9754166
IS_A2_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9754166, upper bound: 5471.9772013
IS_A2_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9772013, upper bound: 5471.9765664
IS_A2_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.16
Output dim: 3, lower bound: -5471.9772013, upper bound: 5471.9783516

## BFS IS instance: IS_A1_B1_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -739.4706421, 3226.6000977, -784.8292847, 3422.3503418, -4161.8208008, 4011.4291992
1: -932.3169556, 3662.2214355, -988.7324219, 3884.1074219, -4816.4243164, 4650.9541016
2: -947.3774414, 3657.3710938, -1005.0864258, 3878.0317383, -4825.4091797, 4662.4565430
3: -1490.3162842, 3805.3459473, -1580.8051758, 4035.8415527, -5526.1577148, 5386.1513672
4: -1493.4432373, 3646.0227051, -1585.2927246, 3864.8032227, -5358.2446289, 5231.3149414

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9776886, upper bound: 5471.9772935
time: 1.07 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9777756, upper bound: 5471.9773697
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -755.5624390, 3293.9741211, -785.1885376, 3423.9638672, -4179.5253906, 4079.1625977
1: -952.1963501, 3738.8527832, -989.1842041, 3885.9379883, -4838.1337891, 4728.0366211
2: -967.6879883, 3732.7165527, -1005.5472412, 3879.8583984, -4847.5463867, 4738.2631836
3: -1522.4324951, 3885.1569824, -1581.5344238, 4037.7421875, -5560.1743164, 5466.6909180
4: -1525.8742676, 3721.0661621, -1586.0260010, 3866.6147461, -5392.4892578, 5307.0917969

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9774441, upper bound: 5471.9775728
time: 1.06 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9774441, upper bound: 5471.9775810
time: 1.21 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -771.3369141, 3360.5925293, -784.8292847, 3422.3503418, -4193.6870117, 4145.4218750
1: -971.5647583, 3814.5788574, -988.7324219, 3884.1074219, -4855.6708984, 4803.3115234
2: -987.7217407, 3808.3706055, -1005.0864258, 3878.0317383, -4865.7534180, 4813.4565430
3: -1553.1181641, 3963.6816406, -1580.8051758, 4035.8415527, -5588.9599609, 5544.4868164
4: -1557.0007324, 3796.2529297, -1585.2927246, 3864.8032227, -5421.8032227, 5381.5454102

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9776886, upper bound: 5471.9777429
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9777760, upper bound: 5471.9778175
time: 1.18 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -786.9946899, 3426.2744141, -785.1885376, 3423.9638672, -4210.9584961, 4211.4614258
1: -990.9156494, 3889.0983887, -989.1842041, 3885.9379883, -4876.8530273, 4878.2827148
2: -1007.4641113, 3881.6296387, -1005.5472412, 3879.8583984, -4887.3208008, 4887.1752930
3: -1584.4553223, 4041.4416504, -1581.5344238, 4037.7421875, -5622.1972656, 5622.9755859
4: -1588.5415039, 3869.2661133, -1586.0260010, 3866.6147461, -5455.1562500, 5455.2919922

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9774509, upper bound: 5471.9777437
time: 1.23 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9774509, upper bound: 5471.9777519
time: 1.17 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -772.2854004, 3366.2541504, -783.9408569, 3416.7700195, -4189.0556641, 4150.1948242
1: -973.2121582, 3820.8588867, -987.1700439, 3877.9267578, -4851.1381836, 4808.0283203
2: -989.0764160, 3814.9716797, -1003.7996826, 3871.5749512, -4860.6513672, 4818.7714844
3: -1555.8546143, 3970.4899902, -1578.2017822, 4029.1918945, -5585.0463867, 5548.6914062
4: -1559.4622803, 3802.8293457, -1582.9591064, 3858.3757324, -5417.8369141, 5385.7866211

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9776753, upper bound: 5471.9779345
time: 1.31 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9777276, upper bound: 5471.9779761
time: 1.16 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -772.6435547, 3367.8562012, -799.5685425, 3482.4748535, -4255.1176758, 4167.4248047
1: -973.6626587, 3822.6748047, -1006.4780273, 3952.4672852, -4926.1298828, 4829.1528320
2: -989.5352173, 3816.7854004, -1023.5108032, 3944.8393555, -4934.3740234, 4840.2949219
3: -1556.5802002, 3972.3762207, -1609.4758301, 4106.9897461, -5663.5688477, 5581.8520508
4: -1560.1916504, 3804.6279297, -1614.4324951, 3931.3928223, -5491.5844727, 5419.0600586

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 44

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9774307, upper bound: 5471.9775894
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9776323, upper bound: 5471.9775894
time: 1.32 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -771.3369141, 3360.5925293, -816.5362549, 3555.7631836, -4327.0991211, 4177.1279297
1: -971.5647583, 3814.5788574, -1027.8046875, 4035.6491699, -5007.2124023, 4842.3837891
2: -987.7217407, 3808.3706055, -1045.2250977, 4028.2463379, -5015.9682617, 4853.5957031
3: -1553.1181641, 3963.6816406, -1643.3934326, 4193.5600586, -5746.6782227, 5607.0747070
4: -1557.0007324, 3796.2529297, -1648.5541992, 4014.3132324, -5571.3134766, 5444.8066406

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9775563, upper bound: 5471.9762312
time: 1.11 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9778380, upper bound: 5471.9778263
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -786.9946899, 3426.2744141, -816.8909912, 3557.3525391, -4344.3471680, 4243.1650391
1: -990.9156494, 3889.0983887, -1028.2508545, 4037.4504395, -5028.3662109, 4917.3491211
2: -1007.4641113, 3881.6296387, -1045.6796875, 4030.0437012, -5037.5073242, 4927.3090820
3: -1584.4553223, 4041.4416504, -1644.1126709, 4195.4311523, -5779.8867188, 5685.5532227
4: -1588.5415039, 3869.2661133, -1649.2773438, 4016.0964355, -5604.6367188, 5518.5434570

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9775080, upper bound: 5471.9777509
time: 1.08 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9775080, upper bound: 5471.9777604
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -836.8504028, 3664.7092285, -783.2211914, 3415.2158203, -4252.0664062, 4447.9301758
1: -1054.7117920, 4159.2958984, -986.7105713, 3875.9697266, -4930.6816406, 5146.0063477
2: -1071.7451172, 4152.1977539, -1003.0382080, 3869.9887695, -4941.7338867, 5155.2353516
3: -1688.7973633, 4319.2319336, -1577.5126953, 4027.4045410, -5716.2011719, 5896.7441406
4: -1690.6484375, 4136.1201172, -1582.0108643, 3856.7988281, -5547.4462891, 5718.1303711

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9693755, upper bound: 5471.9758297
time: 1.07 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9692726, upper bound: 5471.9750866
time: 1.21 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -880.9808960, 3856.3366699, -783.2211914, 3415.2158203, -4296.1967773, 4639.5576172
1: -1109.4549561, 4377.3452148, -986.7105713, 3875.9697266, -4985.4248047, 5364.0556641
2: -1127.7741699, 4368.0615234, -1003.0382080, 3869.9887695, -4997.7622070, 5371.0986328
3: -1776.6966553, 4544.9506836, -1577.5126953, 4027.4045410, -5804.1000977, 6122.4633789
4: -1778.6827393, 4350.6953125, -1582.0108643, 3856.7988281, -5635.4814453, 5932.7041016

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9693754, upper bound: 5471.9780704
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9692726, upper bound: 5471.9774051
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -832.1094971, 3641.4064941, -785.1885376, 3423.9638672, -4256.0732422, 4426.5942383
1: -1048.9846191, 4132.3852539, -989.1842041, 3885.9379883, -4934.9213867, 5121.5688477
2: -1065.8403320, 4126.3354492, -1005.5472412, 3879.8583984, -4945.6977539, 5131.8808594
3: -1679.5003662, 4292.2343750, -1581.5344238, 4037.7421875, -5717.2426758, 5873.7685547
4: -1681.1772461, 4110.7348633, -1586.0260010, 3866.6147461, -5547.7915039, 5696.7602539

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A2_A1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9751432, upper bound: 5471.9765128
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_A1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9739894, upper bound: 5471.9763720
time: 1.10 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -876.1057739, 3831.8208008, -785.1885376, 3423.9638672, -4300.0688477, 4617.0083008
1: -1103.7222900, 4348.3916016, -989.1842041, 3885.9379883, -4989.6591797, 5337.5742188
2: -1121.8182373, 4341.4082031, -1005.5472412, 3879.8583984, -5001.6767578, 5346.9550781
3: -1767.2618408, 4516.3657227, -1581.5344238, 4037.7421875, -5805.0039062, 6097.8999023
4: -1768.7603760, 4325.0527344, -1586.0260010, 3866.6147461, -5635.3750000, 5911.0781250

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A2_A2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9751432, upper bound: 5471.9777566
time: 1.15 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_A2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9739896, upper bound: 5471.9776735
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -836.8504028, 3664.7092285, -814.9169922, 3548.6230469, -4385.4726562, 4479.6259766
1: -1054.7117920, 4159.2958984, -1025.7645264, 4027.5227051, -5082.2343750, 5185.0600586
2: -1071.7451172, 4152.1977539, -1043.1665039, 4020.1958008, -5091.9404297, 5195.3642578
3: -1688.7973633, 4319.2319336, -1640.0700684, 4185.1201172, -5873.9169922, 5959.3017578
4: -1690.6484375, 4136.1201172, -1645.2575684, 4006.2822266, -5696.9291992, 5781.3779297

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9693754, upper bound: 5471.9758297
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9692726, upper bound: 5471.9750866
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -880.9808960, 3856.3366699, -814.9169922, 3548.6230469, -4429.6040039, 4671.2539062
1: -1109.4549561, 4377.3452148, -1025.7645264, 4027.5227051, -5136.9775391, 5403.1098633
2: -1127.7741699, 4368.0615234, -1043.1665039, 4020.1958008, -5147.9692383, 5411.2275391
3: -1776.6966553, 4544.9506836, -1640.0700684, 4185.1201172, -5961.8159180, 6185.0205078
4: -1778.6827393, 4350.6953125, -1645.2575684, 4006.2822266, -5784.9648438, 5995.9526367

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9693755, upper bound: 5471.9780704
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9692726, upper bound: 5471.9774209
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -832.1094971, 3641.4064941, -816.8909912, 3557.3525391, -4389.4609375, 4458.2973633
1: -1048.9846191, 4132.3852539, -1028.2508545, 4037.4504395, -5086.4345703, 5160.6362305
2: -1065.8403320, 4126.3354492, -1045.6796875, 4030.0437012, -5095.8837891, 5172.0146484
3: -1679.5003662, 4292.2343750, -1644.1126709, 4195.4311523, -5874.9306641, 5936.3461914
4: -1681.1772461, 4110.7348633, -1649.2773438, 4016.0964355, -5697.2709961, 5760.0122070

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9740869, upper bound: 5471.9770376
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9739896, upper bound: 5471.9763805
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -876.1057739, 3831.8208008, -816.8909912, 3557.3525391, -4433.4570312, 4648.7119141
1: -1103.7222900, 4348.3916016, -1028.2508545, 4037.4504395, -5141.1723633, 5376.6420898
2: -1121.8182373, 4341.4082031, -1045.6796875, 4030.0437012, -5151.8618164, 5387.0878906
3: -1767.2618408, 4516.3657227, -1644.1126709, 4195.4311523, -5962.6928711, 6160.4775391
4: -1768.7603760, 4325.0527344, -1649.2773438, 4016.0964355, -5784.8559570, 5974.3300781

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A2_A2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9751432, upper bound: 5471.9777651
time: 1.34 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_A2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9739896, upper bound: 5471.9776819
time: 1.19 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -770.6763306, 3359.1083984, -849.3383789, 3720.4421387, -4491.1186523, 4208.4462891
1: -971.1892090, 3812.7119141, -1070.1608887, 4222.1542969, -5193.3427734, 4882.8730469
2: -987.0273438, 3806.9226074, -1087.6759033, 4214.9399414, -5201.9663086, 4894.5981445
3: -1552.5593262, 3962.0441895, -1713.6392822, 4384.2211914, -5936.7802734, 5675.6826172
4: -1556.1773682, 3794.8190918, -1716.3477783, 4197.7519531, -5753.9291992, 5511.1655273

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_A1_B1_B1_A1

### Relational analysis result of IS_A1_B2_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756809, upper bound: 5471.9697962
time: 1.04 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_B1_A2

### Relational analysis result of IS_A1_B2_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9749748, upper bound: 5471.9697215
time: 1.09 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -770.6763306, 3359.1083984, -893.4342651, 3911.8139648, -4682.4902344, 4252.5424805
1: -971.1892090, 3812.7119141, -1124.8735352, 4439.9233398, -5411.1123047, 4937.5854492
2: -987.0273438, 3806.9226074, -1143.6628418, 4430.5473633, -5417.5737305, 4950.5854492
3: -1552.5593262, 3962.0441895, -1801.4903564, 4609.7016602, -6162.2607422, 5763.5346680
4: -1556.1773682, 3794.8190918, -1804.3184814, 4412.1054688, -5968.2827148, 5599.1372070

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_A1_B1_B2_A1

### Relational analysis result of IS_A1_B2_A1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756809, upper bound: 5471.9738216
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_B2_A2

### Relational analysis result of IS_A1_B2_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9749748, upper bound: 5471.9737704
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -772.6435547, 3367.8562012, -844.6287231, 3697.3071289, -4469.9497070, 4212.4848633
1: -973.6626587, 3822.6748047, -1064.4770508, 4195.3935547, -5169.0561523, 4887.1518555
2: -989.5352173, 3816.7854004, -1081.8171387, 4189.2534180, -5178.7885742, 4898.6020508
3: -1556.5802002, 3972.3762207, -1704.4002686, 4357.3984375, -5913.9780273, 5676.7763672
4: -1560.1916504, 3804.6279297, -1706.9498291, 4172.5205078, -5732.7119141, 5511.5771484

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 38

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9768378, upper bound: 5471.9741669
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9762267, upper bound: 5471.9741083
time: 1.01 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -772.6435547, 3367.8562012, -888.5781860, 3887.4604492, -4660.1030273, 4256.4345703
1: -973.6626587, 3822.6748047, -1119.1608887, 4411.1181641, -5384.7807617, 4941.8359375
2: -989.5352173, 3816.7854004, -1137.7296143, 4404.0561523, -5393.5913086, 4954.5141602
3: -1556.5802002, 3972.3762207, -1792.0755615, 4581.2568359, -6137.8364258, 5764.4516602
4: -1560.1916504, 3804.6279297, -1794.4381104, 4386.5810547, -5946.7724609, 5599.0644531

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_A1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9768378, upper bound: 5471.9759838
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_A1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9762267, upper bound: 5471.9759325
time: 3.73 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -802.3634644, 3492.5703125, -849.3383789, 3720.4421387, -4522.8056641, 4341.9077148
1: -1010.2330933, 3964.3205566, -1070.1608887, 4222.1542969, -5232.3872070, 5034.4814453
2: -1027.1541748, 3957.1511230, -1087.6759033, 4214.9399414, -5242.0932617, 5044.8261719
3: -1615.0947266, 4119.7353516, -1713.6392822, 4384.2211914, -5999.3154297, 5833.3740234
4: -1619.4149170, 3944.3100586, -1716.3477783, 4197.7519531, -5817.1660156, 5660.6572266

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_A2_B1_B1_A1

### Relational analysis result of IS_A1_B2_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756819, upper bound: 5471.9699824
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_B1_A2

### Relational analysis result of IS_A1_B2_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9749748, upper bound: 5471.9699044
time: 1.12 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -802.3634644, 3492.5703125, -893.4342651, 3911.8139648, -4714.1772461, 4386.0043945
1: -1010.2330933, 3964.3205566, -1124.8735352, 4439.9233398, -5450.1562500, 5089.1938477
2: -1027.1541748, 3957.1511230, -1143.6628418, 4430.5473633, -5457.7006836, 5100.8129883
3: -1615.0947266, 4119.7353516, -1801.4903564, 4609.7016602, -6224.7954102, 5921.2255859
4: -1619.4149170, 3944.3100586, -1804.3184814, 4412.1054688, -6031.5200195, 5748.6284180

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_A2_B1_B2_A1

### Relational analysis result of IS_A1_B2_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756819, upper bound: 5471.9739714
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_B2_A2

### Relational analysis result of IS_A1_B2_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9749748, upper bound: 5471.9739111
time: 1.16 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -804.3394775, 3501.3076172, -844.6287231, 3697.3071289, -4501.6459961, 4345.9365234
1: -1012.7216187, 3974.2565918, -1064.4770508, 4195.3935547, -5208.1152344, 5038.7324219
2: -1029.6697998, 3967.0075684, -1081.8171387, 4189.2534180, -5218.9233398, 5048.8242188
3: -1619.1402588, 4130.0576172, -1704.4002686, 4357.3984375, -5976.5385742, 5834.4570312
4: -1623.4382324, 3954.1323242, -1706.9498291, 4172.5205078, -5795.9584961, 5661.0820312

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_A1

### Relational analysis result of IS_A1_B2_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9768383, upper bound: 5471.9743180
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_A2

### Relational analysis result of IS_A1_B2_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9762173, upper bound: 5471.9742664
time: 1.01 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -804.3394775, 3501.3076172, -888.5781860, 3887.4604492, -4691.7993164, 4389.8857422
1: -1012.7216187, 3974.2565918, -1119.1608887, 4411.1181641, -5423.8398438, 5093.4165039
2: -1029.6697998, 3967.0075684, -1137.7296143, 4404.0561523, -5433.7260742, 5104.7363281
3: -1619.1402588, 4130.0576172, -1792.0755615, 4581.2568359, -6200.3969727, 5922.1323242
4: -1623.4382324, 3954.1323242, -1794.4381104, 4386.5810547, -6010.0180664, 5748.5698242

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_A1

### Relational analysis result of IS_A1_B2_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9768383, upper bound: 5471.9761340
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_A2

### Relational analysis result of IS_A1_B2_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9762336, upper bound: 5471.9760817
time: 1.09 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -836.8504028, 3664.7092285, -847.8782349, 3711.4760742, -4548.3266602, 4512.5874023
1: -1054.7117920, 4159.2958984, -1068.5543213, 4211.4501953, -5266.1616211, 5227.8496094
2: -1071.7451172, 4152.1977539, -1085.9305420, 4205.2583008, -5277.0024414, 5238.1279297
3: -1688.7973633, 4319.2319336, -1710.8579102, 4374.0307617, -6062.8271484, 6030.0893555
4: -1690.6484375, 4136.1201172, -1713.4458008, 4188.3178711, -5878.9653320, 5849.5659180

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9684144, upper bound: 5471.9731533
time: 1.04 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9681408, upper bound: 5471.9717062
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -880.9808960, 3856.3366699, -847.8782349, 3711.4760742, -4592.4570312, 4704.2148438
1: -1109.4549561, 4377.3452148, -1068.5543213, 4211.4501953, -5320.9052734, 5445.8994141
2: -1127.7741699, 4368.0615234, -1085.9305420, 4205.2583008, -5333.0312500, 5453.9912109
3: -1776.6966553, 4544.9506836, -1710.8579102, 4374.0307617, -6150.7255859, 6255.8085938
4: -1778.6827393, 4350.6953125, -1713.4458008, 4188.3178711, -5967.0004883, 6064.1401367

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9684144, upper bound: 5471.9753910
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9681408, upper bound: 5471.9739533
time: 1.25 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -832.1094971, 3641.4064941, -849.9623413, 3720.8525391, -4552.9614258, 4491.3686523
1: -1048.9846191, 4132.3852539, -1071.1772461, 4222.1093750, -5271.0937500, 5203.5620117
2: -1065.8403320, 4126.3354492, -1088.5864258, 4215.8339844, -5281.6743164, 5214.9199219
3: -1679.5003662, 4292.2343750, -1715.1385498, 4385.0781250, -6064.5776367, 6007.3725586
4: -1681.1772461, 4110.7348633, -1717.7025146, 4198.8320312, -5880.0083008, 5828.4365234

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A2_A1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9740178, upper bound: 5471.9731096
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_A1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9727864, upper bound: 5471.9729138
time: 1.21 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -876.1057739, 3831.8208008, -849.9623413, 3720.8525391, -4596.9570312, 4681.7827148
1: -1103.7222900, 4348.3916016, -1071.1772461, 4222.1093750, -5325.8315430, 5419.5678711
2: -1121.8182373, 4341.4082031, -1088.5864258, 4215.8339844, -5337.6523438, 5429.9941406
3: -1767.2618408, 4516.3657227, -1715.1385498, 4385.0781250, -6152.3398438, 6231.5043945
4: -1768.7603760, 4325.0527344, -1717.7025146, 4198.8320312, -5967.5922852, 6042.7548828

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A2_A2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9740178, upper bound: 5471.9742983
time: 1.10 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_A2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9727864, upper bound: 5471.9742152
time: 1.07 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.41 seconds
IS_A1_B1_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9776886, upper bound: 5471.9772935
IS_A1_B1_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9777756, upper bound: 5471.9773697
IS_A1_B1_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9774441, upper bound: 5471.9775728
IS_A1_B1_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9774441, upper bound: 5471.9775810
IS_A1_B1_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9776886, upper bound: 5471.9777429
IS_A1_B1_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9777760, upper bound: 5471.9778175
IS_A1_B1_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9774509, upper bound: 5471.9777437
IS_A1_B1_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9774509, upper bound: 5471.9777519
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9776753, upper bound: 5471.9779345
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9777276, upper bound: 5471.9779761
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9774307, upper bound: 5471.9775894
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9776323, upper bound: 5471.9775894
IS_A1_B1_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9775563, upper bound: 5471.9762312
IS_A1_B1_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9778380, upper bound: 5471.9778263
IS_A1_B1_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9775080, upper bound: 5471.9777509
IS_A1_B1_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9775080, upper bound: 5471.9777604
IS_A1_B1_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9693755, upper bound: 5471.9758297
IS_A1_B1_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9692726, upper bound: 5471.9750866
IS_A1_B1_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9693754, upper bound: 5471.9780704
IS_A1_B1_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9692726, upper bound: 5471.9774051
IS_A1_B1_A2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9751432, upper bound: 5471.9765128
IS_A1_B1_A2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9739894, upper bound: 5471.9763720
IS_A1_B1_A2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9751432, upper bound: 5471.9777566
IS_A1_B1_A2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9739896, upper bound: 5471.9776735
IS_A1_B1_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9693754, upper bound: 5471.9758297
IS_A1_B1_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9692726, upper bound: 5471.9750866
IS_A1_B1_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9693755, upper bound: 5471.9780704
IS_A1_B1_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9692726, upper bound: 5471.9774209
IS_A1_B1_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9740869, upper bound: 5471.9770376
IS_A1_B1_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9739896, upper bound: 5471.9763805
IS_A1_B1_A2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9751432, upper bound: 5471.9777651
IS_A1_B1_A2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9739896, upper bound: 5471.9776819
IS_A1_B2_A1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9756809, upper bound: 5471.9697962
IS_A1_B2_A1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9749748, upper bound: 5471.9697215
IS_A1_B2_A1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9756809, upper bound: 5471.9738216
IS_A1_B2_A1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9749748, upper bound: 5471.9737704
IS_A1_B2_A1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9768378, upper bound: 5471.9741669
IS_A1_B2_A1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9762267, upper bound: 5471.9741083
IS_A1_B2_A1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9768378, upper bound: 5471.9759838
IS_A1_B2_A1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9762267, upper bound: 5471.9759325
IS_A1_B2_A1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9756819, upper bound: 5471.9699824
IS_A1_B2_A1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9749748, upper bound: 5471.9699044
IS_A1_B2_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9756819, upper bound: 5471.9739714
IS_A1_B2_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9749748, upper bound: 5471.9739111
IS_A1_B2_A1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9768383, upper bound: 5471.9743180
IS_A1_B2_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9762173, upper bound: 5471.9742664
IS_A1_B2_A1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9768383, upper bound: 5471.9761340
IS_A1_B2_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9762336, upper bound: 5471.9760817
IS_A1_B2_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9684144, upper bound: 5471.9731533
IS_A1_B2_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9681408, upper bound: 5471.9717062
IS_A1_B2_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9684144, upper bound: 5471.9753910
IS_A1_B2_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9681408, upper bound: 5471.9739533
IS_A1_B2_A2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9740178, upper bound: 5471.9731096
IS_A1_B2_A2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9727864, upper bound: 5471.9729138
IS_A1_B2_A2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9740178, upper bound: 5471.9742983
IS_A1_B2_A2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.41
Output dim: 3, lower bound: -5471.9727864, upper bound: 5471.9742152
IS_A1_B2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9709035, upper bound: 5471.9762192
IS_A1_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9709035, upper bound: 5471.9782835
IS_A1_B2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9750717, upper bound: 5471.9771959
IS_A1_B2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9750717, upper bound: 5471.9783739
IS_A2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9781903, upper bound: 5471.9776274
IS_A2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9775242, upper bound: 5471.9775242
IS_A2_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9786486, upper bound: 5471.9780956
IS_A2_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9789737, upper bound: 5471.9789734
IS_A2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9781903, upper bound: 5471.9778664
IS_A2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9775326, upper bound: 5471.9777259
IS_A2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9781903, upper bound: 5471.9778744
IS_A2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9775326, upper bound: 5471.9777343
IS_A2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9787738, upper bound: 5471.9764112
IS_A2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9789669, upper bound: 5471.9765986
IS_A2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9788888, upper bound: 5471.9780123
IS_A2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9790819, upper bound: 5471.9781991
IS_A2_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9769581, upper bound: 5471.9728106
IS_A2_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9769581, upper bound: 5471.9768099
IS_A2_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9779311, upper bound: 5471.9765886
IS_A2_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9779311, upper bound: 5471.9783769
IS_A2_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9726361, upper bound: 5471.9769581
IS_A2_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9726361, upper bound: 5471.9769582
IS_A2_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9766611, upper bound: 5471.9789701
IS_A2_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9766611, upper bound: 5471.9789701
IS_A2_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9764410, upper bound: 5471.9779311
IS_A2_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9764410, upper bound: 5471.9779311
IS_A2_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9782262, upper bound: 5471.9790814
IS_A2_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9782262, upper bound: 5471.9790815
IS_A2_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9716441, upper bound: 5471.9744436
IS_A2_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9716441, upper bound: 5471.9762283
IS_A2_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9756362, upper bound: 5471.9764555
IS_A2_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9756362, upper bound: 5471.9782402
IS_A2_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9754166, upper bound: 5471.9754166
IS_A2_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9754166, upper bound: 5471.9772013
IS_A2_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9772013, upper bound: 5471.9765664
IS_A2_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9772013, upper bound: 5471.9783516
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=6162.86279296875
rel_dist={3: [-5471.979332446701, 5471.979332446701]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9788631, upper bound: 5471.9789236
time: 1.20 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9789244, upper bound: 5471.9789244
time: 1.08 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.46 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.46
Output dim: 3, lower bound: -5471.9788631, upper bound: 5471.9789236
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.46
Output dim: 3, lower bound: -5471.9789244, upper bound: 5471.9789244

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -850.7169800, 3697.8376465, -863.1370239, 3753.3532715, -4604.0693359, 4560.9746094
1: -1071.5225830, 4196.7158203, -1086.8886719, 4259.3261719, -5330.8486328, 5283.6035156
2: -1089.2393799, 4190.8291016, -1105.0872803, 4253.2548828, -5342.4931641, 5295.9160156
3: -1711.6445312, 4361.7553711, -1736.3594971, 4426.5043945, -6138.1489258, 6098.1137695
4: -1716.2598877, 4177.3193359, -1741.8470459, 4238.6694336, -5954.9291992, 5919.1665039

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781122, upper bound: 5471.9788435
time: 0.94 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781120, upper bound: 5471.9782451
time: 1.02 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -904.5838623, 3936.6628418, -862.7352295, 3751.5666504, -4656.1503906, 4799.3974609
1: -1139.0826416, 4467.9472656, -1086.3846436, 4257.2973633, -5396.3798828, 5554.3315430
2: -1157.8500977, 4461.1088867, -1104.5736084, 4251.2460938, -5409.0961914, 5565.6826172
3: -1819.3218994, 4642.7978516, -1735.5441895, 4424.4018555, -6243.7231445, 6378.3413086
4: -1825.1307373, 4443.3935547, -1741.0228271, 4236.6801758, -6061.8110352, 6184.4165039

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9787245, upper bound: 5471.9788360
time: 1.19 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9789244, upper bound: 5471.9789244
time: 1.00 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.09 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.09
Output dim: 3, lower bound: -5471.9781122, upper bound: 5471.9788435
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.09
Output dim: 3, lower bound: -5471.9781120, upper bound: 5471.9782451
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 4.09
Output dim: 3, lower bound: -5471.9787245, upper bound: 5471.9788360
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 4.09
Output dim: 3, lower bound: -5471.9789244, upper bound: 5471.9789244

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -840.9882812, 3656.5764160, -822.2101440, 3580.1765137, -4421.1645508, 4478.7866211
1: -1059.1828613, 4150.0434570, -1034.9815674, 4063.3266602, -5122.5097656, 5185.0249023
2: -1076.7432861, 4143.8061523, -1052.5050049, 4056.0151367, -5132.7583008, 5196.3105469
3: -1692.2280273, 4313.1806641, -1654.7614746, 4222.3891602, -5914.6162109, 5967.9414062
4: -1696.8093262, 4130.4277344, -1659.9338379, 4042.0654297, -5738.8750000, 5790.3613281

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781120, upper bound: 5471.9782451
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781120, upper bound: 5471.9782451
time: 0.99 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -833.3477173, 3627.4426270, -900.7853394, 3941.4938965, -4774.8413086, 4528.2280273
1: -1049.6213379, 4116.4609375, -1134.5142822, 4472.5200195, -5522.1416016, 5250.9750977
2: -1066.9959717, 4110.7739258, -1153.3151855, 4465.1274414, -5532.1230469, 5264.0878906
3: -1677.5461426, 4277.7397461, -1816.7711182, 4644.8984375, -6322.4443359, 6094.5107422
4: -1681.7045898, 4096.6748047, -1819.2148438, 4447.2148438, -6128.9194336, 5915.8896484

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781120, upper bound: 5471.9782451
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781120, upper bound: 5471.9782451
time: 1.04 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -877.9072876, 3814.6838379, -861.9840698, 3748.1040039, -4626.0112305, 4676.6679688
1: -1105.8774414, 4330.1918945, -1085.4488525, 4253.3745117, -5359.2519531, 5415.6401367
2: -1124.1602783, 4324.1318359, -1103.6260986, 4247.3544922, -5371.5136719, 5427.7578125
3: -1766.0141602, 4500.7124023, -1734.0335693, 4420.3535156, -6186.3666992, 6234.7460938
4: -1771.0197754, 4310.4692383, -1739.4920654, 4232.8945312, -6003.9140625, 6049.9614258

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_A1_A1

### Relational analysis result of IS_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9786410, upper bound: 5471.9780475
time: 1.02 seconds

## Relational analysis of IS_A2_A1_A2

### Relational analysis result of IS_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780171, upper bound: 5471.9780475
time: 1.04 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -904.4071655, 3935.9562988, -862.7352295, 3751.5666504, -4655.9736328, 4798.6914062
1: -1138.8594971, 4467.1440430, -1086.3846436, 4257.2973633, -5396.1567383, 5553.5283203
2: -1157.6224365, 4460.3022461, -1104.5736084, 4251.2460938, -5408.8686523, 5564.8754883
3: -1818.9764404, 4641.9545898, -1735.5441895, 4424.4018555, -6243.3779297, 6377.4980469
4: -1824.7785645, 4442.5830078, -1741.0228271, 4236.6801758, -6061.4589844, 6183.6054688

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_A2_A1

### Relational analysis result of IS_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9788421, upper bound: 5471.9782085
time: 1.03 seconds

## Relational analysis of IS_A2_A2_A2

### Relational analysis result of IS_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782085, upper bound: 5471.9782085
time: 1.36 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.29 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.29
Output dim: 3, lower bound: -5471.9781120, upper bound: 5471.9782451
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.29
Output dim: 3, lower bound: -5471.9781120, upper bound: 5471.9782451
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.29
Output dim: 3, lower bound: -5471.9781120, upper bound: 5471.9782451
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.29
Output dim: 3, lower bound: -5471.9781120, upper bound: 5471.9782451
IS_A2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 4.29
Output dim: 3, lower bound: -5471.9786410, upper bound: 5471.9780475
IS_A2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 4.29
Output dim: 3, lower bound: -5471.9780171, upper bound: 5471.9780475
IS_A2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 4.29
Output dim: 3, lower bound: -5471.9788421, upper bound: 5471.9782085
IS_A2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 4.29
Output dim: 3, lower bound: -5471.9782085, upper bound: 5471.9782085

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -809.6653442, 3524.1467285, -822.2101440, 3580.1765137, -4389.8417969, 4346.3569336
1: -1019.4607544, 4000.1494141, -1034.9815674, 4063.3266602, -5082.7875977, 5035.1308594
2: -1036.5018311, 3992.9978027, -1052.5050049, 4056.0151367, -5092.5170898, 5045.5019531
3: -1629.8031006, 4157.0307617, -1654.7614746, 4222.3891602, -5852.1918945, 5811.7919922
4: -1634.1029053, 3980.1218262, -1659.9338379, 4042.0654297, -5676.1684570, 5640.0551758

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9778382, upper bound: 5471.9788379
time: 1.14 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781122, upper bound: 5471.9788402
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -888.3382568, 3885.9843750, -822.2101440, 3580.1765137, -4468.5146484, 4708.1943359
1: -1119.1064453, 4409.9409180, -1034.9815674, 4063.3266602, -5182.4331055, 5444.9213867
2: -1137.4340820, 4402.6269531, -1052.5050049, 4056.0151367, -5193.4492188, 5455.1313477
3: -1792.0124512, 4580.1577148, -1654.7614746, 4222.3891602, -6014.4013672, 6234.9189453
4: -1793.5903320, 4385.8295898, -1659.9338379, 4042.0654297, -5835.6557617, 6045.7636719

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9778382, upper bound: 5471.9788379
time: 1.14 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781122, upper bound: 5471.9788402
time: 1.04 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -809.6653442, 3524.1467285, -900.7853394, 3941.4938965, -4751.1591797, 4424.9321289
1: -1019.4607544, 4000.1494141, -1134.5142822, 4472.5200195, -5491.9809570, 5134.6635742
2: -1036.5018311, 3992.9978027, -1153.3151855, 4465.1274414, -5501.6293945, 5146.3125000
3: -1629.8031006, 4157.0307617, -1816.7711182, 4644.8984375, -6274.7016602, 5973.8017578
4: -1634.1029053, 3980.1218262, -1819.2148438, 4447.2148438, -6081.3178711, 5799.3354492

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781120, upper bound: 5471.9781179
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781120, upper bound: 5471.9782451
time: 1.13 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -888.3382568, 3885.9843750, -900.7853394, 3941.4938965, -4829.8320312, 4786.7695312
1: -1119.1064453, 4409.9409180, -1134.5142822, 4472.5200195, -5591.6264648, 5544.4545898
2: -1137.4340820, 4402.6269531, -1153.3151855, 4465.1274414, -5602.5615234, 5555.9423828
3: -1792.0124512, 4580.1577148, -1816.7711182, 4644.8984375, -6436.9106445, 6396.9287109
4: -1793.5903320, 4385.8295898, -1819.2148438, 4447.2148438, -6240.8046875, 6205.0444336

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9767793, upper bound: 5471.9763536
time: 1.21 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781120, upper bound: 5471.9782451
time: 1.09 seconds

## BFS IS instance: IS_A2_A1_A1

### Backsubstitution after applying IS history:
0: -836.8653564, 3641.1677246, -852.2792969, 3706.9541016, -4543.8183594, 4493.4467773
1: -1053.7990723, 4133.8930664, -1073.1386719, 4206.8291016, -5260.6279297, 5207.0312500
2: -1071.4710693, 4126.4687500, -1091.1611328, 4200.4633789, -5271.9345703, 5217.6289062
3: -1684.1738281, 4296.1738281, -1714.6685791, 4371.9047852, -6056.0776367, 6010.8422852
4: -1688.9489746, 4113.4960938, -1720.0938721, 4186.1362305, -5875.0849609, 5833.5888672

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_A1_A1_B1

### Relational analysis result of IS_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780171, upper bound: 5471.9780475
time: 1.09 seconds

## Relational analysis of IS_A2_A1_A1_B2

### Relational analysis result of IS_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780171, upper bound: 5471.9780475
time: 1.22 seconds

## BFS IS instance: IS_A2_A1_A2

### Backsubstitution after applying IS history:
0: -915.8217163, 4003.0617676, -844.6572266, 3677.8522949, -4593.6723633, 4847.7187500
1: -1153.8229980, 4543.6396484, -1063.5947266, 4173.2636719, -5327.0869141, 5607.2343750
2: -1172.8452148, 4536.1694336, -1081.4279785, 4167.4594727, -5340.3046875, 5617.5966797
3: -1846.8117676, 4719.3300781, -1700.0083008, 4336.4931641, -6183.3046875, 6419.3374023
4: -1848.9246826, 4519.6440430, -1705.0091553, 4152.4038086, -6001.3276367, 6224.6533203

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_A1_A2_B1

### Relational analysis result of IS_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780171, upper bound: 5471.9780475
time: 1.06 seconds

## Relational analysis of IS_A2_A1_A2_B2

### Relational analysis result of IS_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780171, upper bound: 5471.9780475
time: 1.17 seconds

## BFS IS instance: IS_A2_A2_A1

### Backsubstitution after applying IS history:
0: -862.8826294, 3760.1909180, -853.0240479, 3710.3886719, -4573.2709961, 4613.2143555
1: -1086.2114258, 4268.2900391, -1074.0664062, 4210.7211914, -5296.9326172, 5342.3564453
2: -1104.3425293, 4260.2314453, -1092.1009521, 4204.3237305, -5308.6660156, 5352.3315430
3: -1736.1597900, 4434.8232422, -1716.1656494, 4375.9213867, -6112.0810547, 6150.9887695
4: -1741.6884766, 4243.2480469, -1721.6114502, 4189.8881836, -5931.5766602, 5964.8579102

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_A2_A1_B1

### Relational analysis result of IS_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782085, upper bound: 5471.9782085
time: 1.17 seconds

## Relational analysis of IS_A2_A2_A1_B2

### Relational analysis result of IS_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782085, upper bound: 5471.9782085
time: 1.16 seconds

## BFS IS instance: IS_A2_A2_A2

### Backsubstitution after applying IS history:
0: -942.9987793, 4127.6245117, -845.4084473, 3681.3164062, -4624.3144531, 4973.0327148
1: -1187.6744385, 4684.4360352, -1064.5328369, 4177.1889648, -5364.8632812, 5748.9687500
2: -1207.1914062, 4676.1494141, -1082.3762207, 4171.3520508, -5378.5434570, 5758.5244141
3: -1901.3382568, 4864.5913086, -1701.5201416, 4340.5429688, -6241.8813477, 6566.1108398
4: -1904.2214355, 4655.4824219, -1706.5394287, 4156.1884766, -6060.4086914, 6362.0209961

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_A2_A2_B1

### Relational analysis result of IS_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782085, upper bound: 5471.9782085
time: 1.09 seconds

## Relational analysis of IS_A2_A2_A2_B2

### Relational analysis result of IS_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782085, upper bound: 5471.9782085
time: 0.99 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.09 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -5471.9778382, upper bound: 5471.9788379
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -5471.9781122, upper bound: 5471.9788402
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -5471.9778382, upper bound: 5471.9788379
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -5471.9781122, upper bound: 5471.9788402
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -5471.9781120, upper bound: 5471.9781179
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -5471.9781120, upper bound: 5471.9782451
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -5471.9767793, upper bound: 5471.9763536
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -5471.9781120, upper bound: 5471.9782451
IS_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -5471.9780171, upper bound: 5471.9780475
IS_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -5471.9780171, upper bound: 5471.9780475
IS_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -5471.9780171, upper bound: 5471.9780475
IS_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -5471.9780171, upper bound: 5471.9780475
IS_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -5471.9782085, upper bound: 5471.9782085
IS_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -5471.9782085, upper bound: 5471.9782085
IS_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -5471.9782085, upper bound: 5471.9782085
IS_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.09
Output dim: 3, lower bound: -5471.9782085, upper bound: 5471.9782085

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -802.0603638, 3490.9555664, -785.1885376, 3423.9638672, -4226.0244141, 4276.1425781
1: -1009.9724731, 3962.5041504, -989.1842041, 3885.9379883, -4895.9096680, 4951.6870117
2: -1026.7833252, 3955.4987793, -1005.5472412, 3879.8583984, -4906.6416016, 4961.0449219
3: -1614.6413574, 4118.0410156, -1581.5344238, 4037.7421875, -5652.3837891, 5699.5751953
4: -1618.7797852, 3942.9968262, -1586.0260010, 3866.6147461, -5485.3945312, 5529.0224609

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9786252, upper bound: 5471.9786978
time: 1.15 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9786252, upper bound: 5471.9789225
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -807.4489746, 3514.5939941, -816.8909912, 3557.3525391, -4364.8012695, 4331.4848633
1: -1016.6495361, 3989.3198242, -1028.2508545, 4037.4504395, -5054.1000977, 5017.5708008
2: -1033.6594238, 3982.1347656, -1045.6796875, 4030.0437012, -5063.7031250, 5027.8144531
3: -1625.3428955, 4145.7500000, -1644.1126709, 4195.4311523, -5820.7739258, 5789.8623047
4: -1629.6617432, 3969.2585449, -1649.2773438, 4016.0964355, -5645.7573242, 5618.5361328

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9788621, upper bound: 5471.9786978
time: 1.19 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9788621, upper bound: 5471.9789225
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -880.1518555, 3850.8151855, -785.1885376, 3423.9638672, -4304.1157227, 4636.0019531
1: -1108.8624268, 4370.0600586, -989.1842041, 3885.9379883, -4994.7998047, 5359.2431641
2: -1126.9637451, 4362.7934570, -1005.5472412, 3879.8583984, -5006.8217773, 5368.3403320
3: -1775.7862549, 4538.8017578, -1581.5344238, 4037.7421875, -5813.5283203, 6120.3359375
4: -1777.1639404, 4346.3125000, -1586.0260010, 3866.6147461, -5643.7788086, 5932.3383789

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756448, upper bound: 5471.9775248
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756448, upper bound: 5471.9788379
time: 1.18 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -886.0529785, 3875.8107910, -816.8909912, 3557.3525391, -4443.4052734, 4692.7011719
1: -1116.2292480, 4398.3701172, -1028.2508545, 4037.4504395, -5153.6796875, 5426.6210938
2: -1134.5056152, 4391.1220703, -1045.6796875, 4030.0437012, -5164.5493164, 5436.8017578
3: -1787.3645020, 4568.1596680, -1644.1126709, 4195.4311523, -5982.7954102, 6212.2714844
4: -1788.9238281, 4374.3876953, -1649.2773438, 4016.0964355, -5805.0190430, 6023.6650391

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9760177, upper bound: 5471.9775248
time: 1.08 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9760177, upper bound: 5471.9788402
time: 1.04 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -809.6653442, 3524.1467285, -888.3382568, 3885.9843750, -4695.6494141, 4412.4843750
1: -1019.4607544, 4000.1494141, -1119.1064453, 4409.9409180, -5429.4018555, 5119.2558594
2: -1036.5018311, 3992.9978027, -1137.4340820, 4402.6269531, -5439.1289062, 5130.4311523
3: -1629.8031006, 4157.0307617, -1792.0124512, 4580.1577148, -6209.9609375, 5949.0429688
4: -1634.1029053, 3980.1218262, -1793.5903320, 4385.8295898, -6019.9326172, 5773.7114258

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784116, upper bound: 5471.9764689
time: 1.06 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2

### Relational analysis result of IS_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9787257, upper bound: 5471.9780861
time: 1.08 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -809.6653442, 3524.1467285, -943.1829224, 4128.3657227, -4938.0302734, 4467.3291016
1: -1019.4607544, 4000.1494141, -1187.9083252, 4685.2797852, -5704.7407227, 5188.0576172
2: -1036.5018311, 3992.9978027, -1207.4289551, 4676.9970703, -5713.4990234, 5200.4267578
3: -1629.8031006, 4157.0307617, -1901.7006836, 4865.4750977, -6495.2783203, 6058.7314453
4: -1634.1029053, 3980.1218262, -1904.5893555, 4656.3349609, -6290.4379883, 5884.7109375

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784116, upper bound: 5471.9766914
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2

### Relational analysis result of IS_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9787257, upper bound: 5471.9782161
time: 1.20 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -880.1518555, 3850.8151855, -849.9623413, 3720.8525391, -4601.0039062, 4700.7763672
1: -1108.8624268, 4370.0600586, -1071.1772461, 4222.1093750, -5330.9716797, 5441.2363281
2: -1126.9637451, 4362.7934570, -1088.5864258, 4215.8339844, -5342.7978516, 5451.3793945
3: -1775.7862549, 4538.8017578, -1715.1385498, 4385.0781250, -6160.8642578, 6253.9399414
4: -1777.1639404, 4346.3125000, -1717.7025146, 4198.8320312, -5975.9956055, 6064.0146484

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9748663, upper bound: 5471.9751947
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9748663, upper bound: 5471.9763536
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -886.0529785, 3875.8107910, -893.6677856, 3909.9631348, -4796.0161133, 4769.4785156
1: -1116.2292480, 4398.3701172, -1125.5454102, 4436.6674805, -5552.8964844, 5523.9155273
2: -1134.5056152, 4391.1220703, -1144.2006836, 4429.4516602, -5563.9565430, 5535.3212891
3: -1787.3645020, 4568.1596680, -1802.3121338, 4607.6967773, -6395.0610352, 6370.4716797
4: -1788.9238281, 4374.3876953, -1804.7126465, 4411.7182617, -6200.6420898, 6179.1000977

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9760177, upper bound: 5471.9768682
time: 1.26 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9760177, upper bound: 5471.9782451
time: 1.18 seconds

## BFS IS instance: IS_A2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -836.8653564, 3641.1677246, -821.0838623, 3575.0615234, -4411.9262695, 4462.2514648
1: -1053.7990723, 4133.8930664, -1033.5743408, 4057.5246582, -5111.3237305, 5167.4663086
2: -1071.4710693, 4126.4687500, -1051.0756836, 4050.2685547, -5121.7387695, 5177.5439453
3: -1684.1738281, 4296.1738281, -1652.4906006, 4216.3901367, -5900.5634766, 5948.6645508
4: -1688.9489746, 4113.4960938, -1657.6346436, 4036.4411621, -5725.3896484, 5771.1308594

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_A1_A1_B1_A1

### Relational analysis result of IS_A2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9786270, upper bound: 5471.9777512
time: 1.06 seconds

## Relational analysis of IS_A2_A1_A1_B1_A2

### Relational analysis result of IS_A2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9786366, upper bound: 5471.9780475
time: 1.06 seconds

## BFS IS instance: IS_A2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -836.8653564, 3641.1677246, -899.5542603, 3935.9235840, -4772.7885742, 4540.7211914
1: -1053.7990723, 4133.8930664, -1132.9741211, 4466.2016602, -5520.0009766, 5266.8666992
2: -1071.4710693, 4126.4687500, -1151.7558594, 4458.8608398, -5530.3320312, 5278.2236328
3: -1684.1738281, 4296.1738281, -1814.2869873, 4638.3657227, -6322.5395508, 6110.4609375
4: -1688.9489746, 4113.4960938, -1816.7099609, 4441.0703125, -6130.0185547, 5930.2055664

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_A1_B2_B1

### Relational analysis result of IS_A2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784773, upper bound: 5471.9776668
time: 1.14 seconds

## Relational analysis of IS_A2_A1_A1_B2_B2

### Relational analysis result of IS_A2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9786410, upper bound: 5471.9780475
time: 1.07 seconds

## BFS IS instance: IS_A2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -915.8217163, 4003.0617676, -821.0838623, 3575.0615234, -4490.8828125, 4824.1455078
1: -1153.8229980, 4543.6396484, -1033.5743408, 4057.5246582, -5211.3476562, 5577.2133789
2: -1172.8452148, 4536.1694336, -1051.0756836, 4050.2685547, -5223.1127930, 5587.2446289
3: -1846.8117676, 4719.3300781, -1652.4906006, 4216.3901367, -6063.2011719, 6371.8203125
4: -1848.9246826, 4519.6440430, -1657.6346436, 4036.4411621, -5885.3652344, 6177.2788086

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_A1_A2_B1_A1

### Relational analysis result of IS_A2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9765305, upper bound: 5471.9778351
time: 1.00 seconds

## Relational analysis of IS_A2_A1_A2_B1_A2

### Relational analysis result of IS_A2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9779735, upper bound: 5471.9779530
time: 1.17 seconds

## BFS IS instance: IS_A2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -915.8217163, 4003.0617676, -899.5542603, 3935.9235840, -4851.7446289, 4902.6152344
1: -1153.8229980, 4543.6396484, -1132.9741211, 4466.2016602, -5620.0244141, 5676.6137695
2: -1172.8452148, 4536.1694336, -1151.7558594, 4458.8608398, -5631.7060547, 5687.9248047
3: -1846.8117676, 4719.3300781, -1814.2869873, 4638.3657227, -6485.1772461, 6533.6166992
4: -1848.9246826, 4519.6440430, -1816.7099609, 4441.0703125, -6289.9941406, 6336.3535156

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_A1_A2_B2_A1

### Relational analysis result of IS_A2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9765305, upper bound: 5471.9778351
time: 1.13 seconds

## Relational analysis of IS_A2_A1_A2_B2_A2

### Relational analysis result of IS_A2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9779735, upper bound: 5471.9779530
time: 0.99 seconds

## BFS IS instance: IS_A2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -862.8826294, 3760.1909180, -821.8059082, 3578.3908691, -4441.2734375, 4581.9956055
1: -1086.2114258, 4268.2900391, -1034.4746094, 4061.2988281, -5147.5102539, 5302.7646484
2: -1104.3425293, 4260.2314453, -1051.9885254, 4054.0080566, -5158.3505859, 5312.2197266
3: -1736.1597900, 4434.8232422, -1653.9433594, 4220.2871094, -5956.4467773, 6088.7666016
4: -1741.6884766, 4243.2480469, -1659.1070557, 4040.0778809, -5781.7666016, 5902.3549805

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_A2_A1_B1_A1

### Relational analysis result of IS_A2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9787569, upper bound: 5471.9779277
time: 1.19 seconds

## Relational analysis of IS_A2_A2_A1_B1_A2

### Relational analysis result of IS_A2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9788387, upper bound: 5471.9782085
time: 1.04 seconds

## BFS IS instance: IS_A2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -862.8826294, 3760.1909180, -900.3892822, 3939.7429199, -4802.6254883, 4660.5791016
1: -1086.2114258, 4268.2900391, -1134.0190430, 4470.5336914, -5556.7451172, 5402.3090820
2: -1104.3425293, 4260.2314453, -1152.8118896, 4463.1577148, -5567.5000000, 5413.0434570
3: -1736.1597900, 4434.8232422, -1815.9688721, 4642.8388672, -6378.9985352, 6250.7919922
4: -1741.6884766, 4243.2480469, -1818.4082031, 4445.2626953, -6186.9506836, 6061.6552734

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_A2_A1_B2_A1

### Relational analysis result of IS_A2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9788361, upper bound: 5471.9779277
time: 1.08 seconds

## Relational analysis of IS_A2_A2_A1_B2_A2

### Relational analysis result of IS_A2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9788387, upper bound: 5471.9782085
time: 1.04 seconds

## BFS IS instance: IS_A2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -942.9987793, 4127.6245117, -821.8059082, 3578.3908691, -4521.3896484, 4949.4306641
1: -1187.6744385, 4684.4360352, -1034.4746094, 4061.2988281, -5248.9731445, 5718.9106445
2: -1207.1914062, 4676.1494141, -1051.9885254, 4054.0080566, -5261.1992188, 5728.1372070
3: -1901.3382568, 4864.5913086, -1653.9433594, 4220.2871094, -6121.6250000, 6518.5341797
4: -1904.2214355, 4655.4824219, -1659.1070557, 4040.0778809, -5944.2988281, 6314.5888672

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_A2_A2_B1_A1

### Relational analysis result of IS_A2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9764538, upper bound: 5471.9780701
time: 0.99 seconds

## Relational analysis of IS_A2_A2_A2_B1_A2

### Relational analysis result of IS_A2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781644, upper bound: 5471.9781644
time: 1.06 seconds

## BFS IS instance: IS_A2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -942.9987793, 4127.6245117, -900.3892822, 3939.7429199, -4882.7416992, 5028.0131836
1: -1187.6744385, 4684.4360352, -1134.0190430, 4470.5336914, -5658.2080078, 5818.4550781
2: -1207.1914062, 4676.1494141, -1152.8118896, 4463.1577148, -5670.3491211, 5828.9609375
3: -1901.3382568, 4864.5913086, -1815.9688721, 4642.8388672, -6544.1772461, 6680.5595703
4: -1904.2214355, 4655.4824219, -1818.4082031, 4445.2626953, -6349.4829102, 6473.8891602

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_A2_A2_B2_A1

### Relational analysis result of IS_A2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9766173, upper bound: 5471.9780701
time: 1.02 seconds

## Relational analysis of IS_A2_A2_A2_B2_A2

### Relational analysis result of IS_A2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781644, upper bound: 5471.9781644
time: 1.31 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.40 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.40
Output dim: 3, lower bound: -5471.9786252, upper bound: 5471.9786978
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.40
Output dim: 3, lower bound: -5471.9786252, upper bound: 5471.9789225
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.40
Output dim: 3, lower bound: -5471.9788621, upper bound: 5471.9786978
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.40
Output dim: 3, lower bound: -5471.9788621, upper bound: 5471.9789225
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.40
Output dim: 3, lower bound: -5471.9756448, upper bound: 5471.9775248
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.40
Output dim: 3, lower bound: -5471.9756448, upper bound: 5471.9788379
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.40
Output dim: 3, lower bound: -5471.9760177, upper bound: 5471.9775248
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.40
Output dim: 3, lower bound: -5471.9760177, upper bound: 5471.9788402
IS_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.40
Output dim: 3, lower bound: -5471.9784116, upper bound: 5471.9764689
IS_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.40
Output dim: 3, lower bound: -5471.9787257, upper bound: 5471.9780861
IS_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.40
Output dim: 3, lower bound: -5471.9784116, upper bound: 5471.9766914
IS_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.40
Output dim: 3, lower bound: -5471.9787257, upper bound: 5471.9782161
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.40
Output dim: 3, lower bound: -5471.9748663, upper bound: 5471.9751947
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.40
Output dim: 3, lower bound: -5471.9748663, upper bound: 5471.9763536
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.40
Output dim: 3, lower bound: -5471.9760177, upper bound: 5471.9768682
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.40
Output dim: 3, lower bound: -5471.9760177, upper bound: 5471.9782451
IS_A2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.40
Output dim: 3, lower bound: -5471.9786270, upper bound: 5471.9777512
IS_A2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.40
Output dim: 3, lower bound: -5471.9786366, upper bound: 5471.9780475
IS_A2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.40
Output dim: 3, lower bound: -5471.9784773, upper bound: 5471.9776668
IS_A2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.40
Output dim: 3, lower bound: -5471.9786410, upper bound: 5471.9780475
IS_A2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.40
Output dim: 3, lower bound: -5471.9765305, upper bound: 5471.9778351
IS_A2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.40
Output dim: 3, lower bound: -5471.9779735, upper bound: 5471.9779530
IS_A2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.40
Output dim: 3, lower bound: -5471.9765305, upper bound: 5471.9778351
IS_A2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.40
Output dim: 3, lower bound: -5471.9779735, upper bound: 5471.9779530
IS_A2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.40
Output dim: 3, lower bound: -5471.9787569, upper bound: 5471.9779277
IS_A2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.40
Output dim: 3, lower bound: -5471.9788387, upper bound: 5471.9782085
IS_A2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.40
Output dim: 3, lower bound: -5471.9788361, upper bound: 5471.9779277
IS_A2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.40
Output dim: 3, lower bound: -5471.9788387, upper bound: 5471.9782085
IS_A2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.40
Output dim: 3, lower bound: -5471.9764538, upper bound: 5471.9780701
IS_A2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.40
Output dim: 3, lower bound: -5471.9781644, upper bound: 5471.9781644
IS_A2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.40
Output dim: 3, lower bound: -5471.9766173, upper bound: 5471.9780701
IS_A2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.40
Output dim: 3, lower bound: -5471.9781644, upper bound: 5471.9781644

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -772.6435547, 3367.8562012, -785.1885376, 3423.9638672, -4196.6069336, 4153.0444336
1: -973.6626587, 3822.6748047, -989.1842041, 3885.9379883, -4859.6000977, 4811.8588867
2: -989.5352173, 3816.7854004, -1005.5472412, 3879.8583984, -4869.3930664, 4822.3320312
3: -1556.5802002, 3972.3762207, -1581.5344238, 4037.7421875, -5594.3222656, 5553.9106445
4: -1560.1916504, 3804.6279297, -1586.0260010, 3866.6147461, -5426.8066406, 5390.6528320

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782030, upper bound: 5471.9778354
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785872, upper bound: 5471.9786978
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -804.3394775, 3501.3076172, -785.1885376, 3423.9638672, -4228.3032227, 4286.4956055
1: -1012.7216187, 3974.2565918, -989.1842041, 3885.9379883, -4898.6591797, 4963.4394531
2: -1029.6697998, 3967.0075684, -1005.5472412, 3879.8583984, -4909.5283203, 4972.5541992
3: -1619.1402588, 4130.0576172, -1581.5344238, 4037.7421875, -5656.8823242, 5711.5917969
4: -1623.4382324, 3954.1323242, -1586.0260010, 3866.6147461, -5490.0527344, 5540.1582031

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782030, upper bound: 5471.9780469
time: 1.05 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785872, upper bound: 5471.9789225
time: 1.34 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -772.6435547, 3367.8562012, -816.8909912, 3557.3525391, -4329.9951172, 4184.7470703
1: -973.6626587, 3822.6748047, -1028.2508545, 4037.4504395, -5011.1132812, 4850.9257812
2: -989.5352173, 3816.7854004, -1045.6796875, 4030.0437012, -5019.5791016, 4862.4648438
3: -1556.5802002, 3972.3762207, -1644.1126709, 4195.4311523, -5752.0112305, 5616.4887695
4: -1560.1916504, 3804.6279297, -1649.2773438, 4016.0964355, -5576.2875977, 5453.9052734

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782030, upper bound: 5471.9778354
time: 1.06 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785872, upper bound: 5471.9786978
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -804.3394775, 3501.3076172, -816.8909912, 3557.3525391, -4361.6914062, 4318.1987305
1: -1012.7216187, 3974.2565918, -1028.2508545, 4037.4504395, -5050.1718750, 5002.5068359
2: -1029.6697998, 3967.0075684, -1045.6796875, 4030.0437012, -5059.7133789, 5012.6875000
3: -1619.1402588, 4130.0576172, -1644.1126709, 4195.4311523, -5814.5712891, 5774.1689453
4: -1623.4382324, 3954.1323242, -1649.2773438, 4016.0964355, -5639.5322266, 5603.4096680

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782030, upper bound: 5471.9780469
time: 1.17 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785872, upper bound: 5471.9789225
time: 1.10 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -837.4635010, 3665.0446777, -785.1885376, 3423.9638672, -4261.4272461, 4450.2324219
1: -1055.7114258, 4159.2089844, -989.1842041, 3885.9379883, -4941.6489258, 5148.3925781
2: -1072.6353760, 4153.0209961, -1005.5472412, 3879.8583984, -4952.4936523, 5158.5673828
3: -1690.2818604, 4320.0239258, -1581.5344238, 4037.7421875, -5728.0239258, 5901.5585938
4: -1691.9719238, 4137.1518555, -1586.0260010, 3866.6147461, -5558.5864258, 5723.1777344

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B1_A1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9712459, upper bound: 5471.9758753
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756296, upper bound: 5471.9775233
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -881.2186890, 3854.4326172, -785.1885376, 3423.9638672, -4305.1826172, 4639.6191406
1: -1110.1368408, 4374.0634766, -989.1842041, 3885.9379883, -4996.0737305, 5363.2470703
2: -1128.3183594, 4366.9267578, -1005.5472412, 3879.8583984, -5008.1762695, 5372.4731445
3: -1777.5480957, 4542.9340820, -1581.5344238, 4037.7421875, -5815.2900391, 6124.4687500
4: -1779.0839844, 4350.3139648, -1586.0260010, 3866.6147461, -5645.6987305, 5936.3398438

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9712459, upper bound: 5471.9781442
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756296, upper bound: 5471.9787745
time: 1.20 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -837.4635010, 3665.0446777, -816.8909912, 3557.3525391, -4394.8159180, 4481.9355469
1: -1055.7114258, 4159.2089844, -1028.2508545, 4037.4504395, -5093.1621094, 5187.4599609
2: -1072.6353760, 4153.0209961, -1045.6796875, 4030.0437012, -5102.6791992, 5198.7006836
3: -1690.2818604, 4320.0239258, -1644.1126709, 4195.4311523, -5885.7128906, 5964.1362305
4: -1691.9719238, 4137.1518555, -1649.2773438, 4016.0964355, -5708.0668945, 5786.4291992

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B2_A1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9712459, upper bound: 5471.9758753
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756296, upper bound: 5471.9775233
time: 1.16 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -881.2186890, 3854.4326172, -816.8909912, 3557.3525391, -4438.5712891, 4671.3227539
1: -1110.1368408, 4374.0634766, -1028.2508545, 4037.4504395, -5147.5874023, 5402.3144531
2: -1128.3183594, 4366.9267578, -1045.6796875, 4030.0437012, -5158.3623047, 5412.6064453
3: -1777.5480957, 4542.9340820, -1644.1126709, 4195.4311523, -5972.9790039, 6187.0458984
4: -1779.0839844, 4350.3139648, -1649.2773438, 4016.0964355, -5795.1791992, 5999.5913086

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B2_A2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9712459, upper bound: 5471.9784699
time: 1.14 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9756296, upper bound: 5471.9787823
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -803.5709229, 3497.1687012, -888.8275146, 3891.2319336, -4694.8022461, 4385.9960938
1: -1011.7813721, 3969.4687500, -1119.3469238, 4417.0312500, -5428.8125000, 5088.8149414
2: -1028.7408447, 3962.5708008, -1137.8276367, 4407.5161133, -5436.2558594, 5100.3984375
3: -1617.3204346, 4125.1616211, -1792.6622314, 4586.1245117, -6203.4433594, 5917.8237305
4: -1621.6857910, 3949.8181152, -1794.6807861, 4389.9921875, -6011.6762695, 5744.4980469

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_B1_B1_B1

### Relational analysis result of IS_A1_B2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757141, upper bound: 5471.9714709
time: 1.13 seconds

## Relational analysis of IS_A1_B2_A1_B1_B1_B2

### Relational analysis result of IS_A1_B2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784121, upper bound: 5471.9764689
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -809.3253174, 3522.6364746, -883.2570190, 3863.5400391, -4672.8652344, 4405.8935547
1: -1019.0331421, 3998.4377441, -1112.7313232, 4384.4619141, -5403.4951172, 5111.1684570
2: -1036.0699463, 3991.2937012, -1130.9761963, 4377.2978516, -5413.3671875, 5122.2700195
3: -1629.1151123, 4155.2573242, -1781.7916260, 4553.7802734, -6182.8950195, 5937.0488281
4: -1633.4172363, 3978.4323730, -1783.3317871, 4360.7568359, -5994.1738281, 5761.7626953

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_B1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9787048, upper bound: 5471.9778503
time: 1.09 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9787248, upper bound: 5471.9780861
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -803.5709229, 3497.1687012, -943.6794434, 4133.3291016, -4936.8999023, 4440.8476562
1: -1011.7813721, 3969.4687500, -1188.1201172, 4691.8979492, -5703.6791992, 5157.5874023
2: -1028.7408447, 3962.5708008, -1207.8543701, 4681.4077148, -5710.1479492, 5170.4252930
3: -1617.3204346, 4125.1616211, -1902.2111816, 4871.1386719, -6488.4580078, 6027.3725586
4: -1621.6857910, 3949.8181152, -1905.7425537, 4659.7895508, -6281.4750977, 5855.5605469

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757649, upper bound: 5471.9722427
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A1_B2_B1_B2

### Relational analysis result of IS_A1_B2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784116, upper bound: 5471.9766914
time: 1.01 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -809.3253174, 3522.6364746, -938.3684082, 4107.0532227, -4916.3784180, 4461.0043945
1: -1019.0331421, 3998.4377441, -1181.8682861, 4661.0883789, -5680.1215820, 5180.3061523
2: -1036.0699463, 3991.2937012, -1201.3096924, 4652.9384766, -5689.0083008, 5192.6035156
3: -1629.1151123, 4155.2573242, -1892.0146484, 4840.4399414, -6469.5546875, 6047.2719727
4: -1633.4172363, 3978.4323730, -1894.8724365, 4632.5317383, -6265.9487305, 5873.3041992

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_B2_B2_B1

### Relational analysis result of IS_A1_B2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9774618, upper bound: 5471.9762770
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2_B2

### Relational analysis result of IS_A1_B2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9787227, upper bound: 5471.9782161
time: 1.14 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -837.4635010, 3665.0446777, -849.9623413, 3720.8525391, -4558.3159180, 4515.0068359
1: -1055.7114258, 4159.2089844, -1071.1772461, 4222.1093750, -5277.8208008, 5230.3857422
2: -1072.6353760, 4153.0209961, -1088.5864258, 4215.8339844, -5288.4692383, 5241.6064453
3: -1690.2818604, 4320.0239258, -1715.1385498, 4385.0781250, -6075.3593750, 6035.1625977
4: -1691.9719238, 4137.1518555, -1717.7025146, 4198.8320312, -5890.8037109, 5854.8544922

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B1_A1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9706856, upper bound: 5471.9742252
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9748663, upper bound: 5471.9751947
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -881.2186890, 3854.4326172, -849.9623413, 3720.8525391, -4602.0712891, 4704.3940430
1: -1110.1368408, 4374.0634766, -1071.1772461, 4222.1093750, -5332.2460938, 5445.2402344
2: -1128.3183594, 4366.9267578, -1088.5864258, 4215.8339844, -5344.1523438, 5455.5122070
3: -1777.5480957, 4542.9340820, -1715.1385498, 4385.0781250, -6162.6254883, 6258.0722656
4: -1779.0839844, 4350.3139648, -1717.7025146, 4198.8320312, -5977.9160156, 6068.0161133

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B1_A2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9706856, upper bound: 5471.9757989
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9748663, upper bound: 5471.9762790
time: 1.01 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -837.4635010, 3665.0446777, -893.6677856, 3909.9631348, -4747.4267578, 4558.7124023
1: -1055.7114258, 4159.2089844, -1125.5454102, 4436.6674805, -5492.3789062, 5284.7543945
2: -1072.6353760, 4153.0209961, -1144.2006836, 4429.4516602, -5502.0869141, 5297.2207031
3: -1690.2818604, 4320.0239258, -1802.3121338, 4607.6967773, -6297.9785156, 6122.3359375
4: -1691.9719238, 4137.1518555, -1804.7126465, 4411.7182617, -6103.6899414, 5941.8642578

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B2_A1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9706856, upper bound: 5471.9754863
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9748662, upper bound: 5471.9768671
time: 1.10 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -881.2186890, 3854.4326172, -893.6677856, 3909.9631348, -4791.1816406, 4748.1000977
1: -1110.1368408, 4374.0634766, -1125.5454102, 4436.6674805, -5546.8041992, 5499.6088867
2: -1128.3183594, 4366.9267578, -1144.2006836, 4429.4516602, -5557.7700195, 5511.1264648
3: -1777.5480957, 4542.9340820, -1802.3121338, 4607.6967773, -6385.2441406, 6345.2460938
4: -1779.0839844, 4350.3139648, -1804.7126465, 4411.7182617, -6190.8022461, 6155.0263672

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9706856, upper bound: 5471.9781152
time: 1.12 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9748663, upper bound: 5471.9782142
time: 1.12 seconds

## BFS IS instance: IS_A2_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -800.8253784, 3489.1635742, -813.5053711, 3541.9794922, -4342.8041992, 4302.6689453
1: -1009.1810303, 3961.0710449, -1024.1187744, 4020.0046387, -5029.1855469, 4985.1884766
2: -1025.7081299, 3954.9877930, -1041.3923340, 4012.8962402, -5038.6044922, 4996.3793945
3: -1612.8914795, 4116.4931641, -1637.3811035, 4177.5292969, -5790.4204102, 5753.8735352
4: -1616.9952393, 3942.5341797, -1642.3675537, 3999.4436035, -5616.4379883, 5584.9018555

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_A1_A1_B1_A1_B1

### Relational analysis result of IS_A2_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784943, upper bound: 5471.9785306
time: 1.13 seconds

## Relational analysis of IS_A2_A1_A1_B1_A1_B2

### Relational analysis result of IS_A2_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784943, upper bound: 5471.9785306
time: 1.06 seconds

## BFS IS instance: IS_A2_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -831.8585815, 3619.5837402, -818.8718872, 3565.5246582, -4397.3823242, 4438.4555664
1: -1047.4550781, 4109.4106445, -1030.7691650, 4046.7104492, -5094.1655273, 5140.1791992
2: -1065.0458984, 4101.9135742, -1048.2393799, 4039.4235840, -5104.4697266, 5150.1523438
3: -1674.1173096, 4270.6875000, -1648.0402832, 4205.1240234, -5879.2412109, 5918.7270508
4: -1678.9276123, 4088.9265137, -1653.2011719, 4025.5993652, -5704.5258789, 5742.1279297

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_A1_A1_B1_A2_B1

### Relational analysis result of IS_A2_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784943, upper bound: 5471.9788344
time: 1.20 seconds

## Relational analysis of IS_A2_A1_A1_B1_A2_B2

### Relational analysis result of IS_A2_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784943, upper bound: 5471.9788344
time: 1.05 seconds

## BFS IS instance: IS_A2_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -836.8653564, 3641.1677246, -886.7242432, 3878.8017578, -4715.6669922, 4527.8916016
1: -1053.7990723, 4133.8930664, -1117.0989990, 4401.9980469, -5455.7968750, 5250.9916992
2: -1071.4710693, 4126.4687500, -1135.4099121, 4394.5625000, -5466.0336914, 5261.8774414
3: -1684.1738281, 4296.1738281, -1788.8558350, 4571.9428711, -6256.1166992, 6085.0297852
4: -1688.9489746, 4113.4960938, -1790.3001709, 4378.1621094, -6067.1098633, 5903.7954102

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_A1_A1_B2_B1_A1

### Relational analysis result of IS_A2_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784794, upper bound: 5471.9773639
time: 1.25 seconds

## Relational analysis of IS_A2_A1_A1_B2_B1_A2

### Relational analysis result of IS_A2_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784794, upper bound: 5471.9776668
time: 1.12 seconds

## BFS IS instance: IS_A2_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -832.6414795, 3622.7861328, -912.0379639, 3991.6496582, -4824.2895508, 4534.8242188
1: -1048.5024414, 4113.0917969, -1148.6245117, 4530.1137695, -5578.6152344, 5261.7148438
2: -1066.0767822, 4105.6557617, -1167.3353271, 4521.9912109, -5588.0678711, 5272.9912109
3: -1675.7363281, 4274.5854492, -1839.4259033, 4704.4477539, -6380.1840820, 6114.0112305
4: -1680.3992920, 4092.9228516, -1841.6347656, 4503.2363281, -6183.6352539, 5934.5566406

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_A1_A1_B2_B2_A1

### Relational analysis result of IS_A2_A1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9786270, upper bound: 5471.9777512
time: 1.14 seconds

## Relational analysis of IS_A2_A1_A1_B2_B2_A2

### Relational analysis result of IS_A2_A1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9786366, upper bound: 5471.9780475
time: 1.10 seconds

## BFS IS instance: IS_A2_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -916.0964355, 4006.4870605, -815.0046387, 3548.1530762, -4464.2495117, 4821.4916992
1: -1153.8082275, 4548.4433594, -1025.9139404, 4026.9182129, -5180.7246094, 5574.3564453
2: -1172.9799805, 4539.1308594, -1043.3321533, 4019.9128418, -5192.8911133, 5582.4628906
3: -1846.9881592, 4723.2812500, -1640.0373535, 4184.5947266, -6031.5830078, 6363.3183594
4: -1849.4436035, 4522.1342773, -1645.2431641, 4006.2067871, -5855.6499023, 6167.3769531

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_A1_A2_B1_A1_A1

### Relational analysis result of IS_A2_A1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9722133, upper bound: 5471.9757766
time: 0.95 seconds

## Relational analysis of IS_A2_A1_A2_B1_A1_A2

### Relational analysis result of IS_A2_A1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9765331, upper bound: 5471.9782903
time: 1.80 seconds

## BFS IS instance: IS_A2_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -911.4382324, 3983.6218262, -820.7448120, 3573.5571289, -4484.9946289, 4804.3666992
1: -1148.3144531, 4521.5424805, -1033.1480713, 4055.8178711, -5204.1323242, 5554.6894531
2: -1167.2752686, 4514.2280273, -1050.6453857, 4048.5698242, -5215.8452148, 5564.8725586
3: -1837.9632568, 4696.4526367, -1651.8045654, 4214.6215820, -6052.5830078, 6348.2568359
4: -1840.0678711, 4497.9003906, -1656.9512939, 4034.7578125, -5874.8256836, 6154.8515625

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_A1_A2_B1_A2_A1

### Relational analysis result of IS_A2_A1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9760855, upper bound: 5471.9773924
time: 1.14 seconds

## Relational analysis of IS_A2_A1_A2_B1_A2_A2

### Relational analysis result of IS_A2_A1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9779820, upper bound: 5471.9786670
time: 1.15 seconds

## BFS IS instance: IS_A2_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -916.0964355, 4006.4870605, -892.6268921, 3904.9028320, -4820.9990234, 4899.1137695
1: -1153.8082275, 4548.4433594, -1124.2456055, 4430.9570312, -5584.7641602, 5672.6884766
2: -1172.9799805, 4539.1308594, -1142.9364014, 4423.8583984, -5596.8378906, 5682.0673828
3: -1846.9881592, 4723.2812500, -1800.0463867, 4601.7919922, -6448.7802734, 6523.3276367
4: -1849.4436035, 4522.1342773, -1802.5886230, 4406.2441406, -6255.6875000, 6324.7226562

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_A1_A2_B2_A1_B1

### Relational analysis result of IS_A2_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9764389, upper bound: 5471.9762880
time: 1.07 seconds

## Relational analysis of IS_A2_A1_A2_B2_A1_B2

### Relational analysis result of IS_A2_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9764389, upper bound: 5471.9778351
time: 1.10 seconds

## BFS IS instance: IS_A2_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -911.4382324, 3983.6218262, -899.2280884, 3934.4807129, -4845.9189453, 4882.8500977
1: -1148.3144531, 4521.5424805, -1132.5646973, 4464.5625000, -5612.8769531, 5654.1064453
2: -1167.2752686, 4514.2280273, -1151.3410645, 4457.2309570, -5624.5063477, 5665.5688477
3: -1837.9632568, 4696.4526367, -1813.6307373, 4636.6689453, -6474.6308594, 6510.0830078
4: -1840.0678711, 4497.9003906, -1816.0510254, 4439.4570312, -6279.5249023, 6313.9511719

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_A1_A2_B2_A2_B1

### Relational analysis result of IS_A2_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9765280, upper bound: 5471.9762889
time: 0.97 seconds

## Relational analysis of IS_A2_A1_A2_B2_A2_B2

### Relational analysis result of IS_A2_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9765280, upper bound: 5471.9779530
time: 0.98 seconds

## BFS IS instance: IS_A2_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -825.6823120, 3603.2805176, -814.2138672, 3545.2490234, -4370.9311523, 4417.4936523
1: -1040.2266846, 4089.9826660, -1025.0024414, 4023.7126465, -5063.9394531, 5114.9843750
2: -1057.0905762, 4083.1723633, -1042.2877197, 4016.5698242, -5073.6596680, 5125.4589844
3: -1662.7667236, 4249.4833984, -1638.8084717, 4181.3598633, -5844.1254883, 5888.2915039
4: -1667.3420410, 4066.9770508, -1643.8129883, 4003.0158691, -5670.3579102, 5710.7895508

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_A2_A1_B1_A1_B1

### Relational analysis result of IS_A2_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9786828, upper bound: 5471.9786828
time: 4.12 seconds

## Relational analysis of IS_A2_A2_A1_B1_A1_B2

### Relational analysis result of IS_A2_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9786828, upper bound: 5471.9786828
time: 0.97 seconds

## BFS IS instance: IS_A2_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -857.5404663, 3737.1198730, -819.5933838, 3568.8544922, -4426.3950195, 4556.7133789
1: -1079.4433594, 4242.1372070, -1031.6693115, 4050.4855957, -5129.9287109, 5273.8066406
2: -1097.4774170, 4233.9843750, -1049.1507568, 4043.1640625, -5140.6411133, 5283.1352539
3: -1725.4224854, 4407.6010742, -1649.4926758, 4209.0219727, -5934.4443359, 6057.0932617
4: -1730.9700928, 4217.0136719, -1654.6719971, 4029.2353516, -5760.2041016, 5871.6850586

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_A2_A1_B1_A2_B1

### Relational analysis result of IS_A2_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9786828, upper bound: 5471.9789233
time: 1.45 seconds

## Relational analysis of IS_A2_A2_A1_B1_A2_B2

### Relational analysis result of IS_A2_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9786828, upper bound: 5471.9789233
time: 1.08 seconds

## BFS IS instance: IS_A2_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -825.6823120, 3603.2805176, -892.1995850, 3904.5727539, -4730.2548828, 4495.4790039
1: -1040.2266846, 4089.9826660, -1123.7733154, 4430.6494141, -5470.8759766, 5213.7558594
2: -1057.0905762, 4083.1723633, -1142.3354492, 4423.3295898, -5480.4199219, 5225.5058594
3: -1662.7667236, 4249.4833984, -1799.7420654, 4601.4770508, -6264.2431641, 6049.2250977
4: -1667.3420410, 4066.9770508, -1801.9803467, 4405.7495117, -6073.0917969, 5868.9570312

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 34

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_A2_A1_B2_A1_B1

### Relational analysis result of IS_A2_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9775231, upper bound: 5471.9759738
time: 1.02 seconds

## Relational analysis of IS_A2_A2_A1_B2_A1_B2

### Relational analysis result of IS_A2_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9775229, upper bound: 5471.9779276
time: 1.20 seconds

## BFS IS instance: IS_A2_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -857.5404663, 3737.1198730, -898.1039429, 3929.5759277, -4787.1162109, 4635.2236328
1: -1079.4433594, 4242.1372070, -1131.1411133, 4458.9697266, -5538.4125977, 5373.2783203
2: -1097.4774170, 4233.9843750, -1149.8820801, 4451.6591797, -5549.1367188, 5383.8662109
3: -1725.4224854, 4407.6010742, -1811.3217773, 4630.8452148, -6356.2675781, 6218.9223633
4: -1730.9700928, 4217.0136719, -1813.7403564, 4433.8266602, -6164.7954102, 6030.7524414

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_A2_A1_B2_A2_B1

### Relational analysis result of IS_A2_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9775231, upper bound: 5471.9763305
time: 1.00 seconds

## Relational analysis of IS_A2_A2_A1_B2_A2_B2

### Relational analysis result of IS_A2_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9775229, upper bound: 5471.9782085
time: 1.36 seconds

## BFS IS instance: IS_A2_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -943.4970093, 4132.5961914, -815.7223511, 3551.4653320, -4494.9624023, 4948.3183594
1: -1187.8886719, 4691.0644531, -1026.8094482, 4030.6713867, -5218.5600586, 5717.8740234
2: -1207.6195068, 4680.5698242, -1044.2397461, 4023.6333008, -5231.2519531, 5724.8085938
3: -1901.8521729, 4870.2656250, -1641.4822998, 4188.4716797, -6090.3237305, 6511.7480469
4: -1905.3791504, 4658.9482422, -1646.7075195, 4009.8229980, -5915.2011719, 6305.6552734

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_A2_A2_B1_A1_A1

### Relational analysis result of IS_A2_A2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9721684, upper bound: 5471.9759038
time: 1.01 seconds

## Relational analysis of IS_A2_A2_A2_B1_A1_A2

### Relational analysis result of IS_A2_A2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9766199, upper bound: 5471.9784678
time: 1.02 seconds

## BFS IS instance: IS_A2_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -938.1854858, 4106.3183594, -821.4664917, 3576.8833008, -4515.0688477, 4927.7846680
1: -1181.6363525, 4660.2524414, -1034.0478516, 4059.5888672, -5241.2250977, 5694.2993164
2: -1201.0739746, 4652.0981445, -1051.5570068, 4052.3071289, -5253.3808594, 5703.6542969
3: -1891.6547852, 4839.5629883, -1653.2563477, 4218.5156250, -6110.1699219, 6492.8193359
4: -1894.5074463, 4631.6870117, -1658.4227295, 4038.3920898, -5932.8984375, 6290.1093750

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_A2_A2_B1_A2_A1

### Relational analysis result of IS_A2_A2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9762531, upper bound: 5471.9775214
time: 1.16 seconds

## Relational analysis of IS_A2_A2_A2_B1_A2_A2

### Relational analysis result of IS_A2_A2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781696, upper bound: 5471.9787809
time: 1.05 seconds

## BFS IS instance: IS_A2_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -943.4970093, 4132.5961914, -893.4566040, 3908.6967773, -4852.1933594, 5026.0527344
1: -1187.8886719, 4691.0644531, -1125.2838135, 4435.2578125, -5623.1464844, 5816.3481445
2: -1207.6195068, 4680.5698242, -1143.9854736, 4428.1274414, -5635.7470703, 5824.5546875
3: -1901.8521729, 4870.2656250, -1801.7180176, 4606.2348633, -6508.0869141, 6671.9833984
4: -1905.3791504, 4658.9482422, -1804.2760010, 4410.4072266, -6315.7861328, 6463.2241211

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_A2_A2_B2_A1_B1

### Relational analysis result of IS_A2_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9765279, upper bound: 5471.9765288
time: 1.22 seconds

## Relational analysis of IS_A2_A2_A2_B2_A1_B2

### Relational analysis result of IS_A2_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9765279, upper bound: 5471.9780700
time: 1.06 seconds

## BFS IS instance: IS_A2_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -938.1854858, 4106.3183594, -900.0626221, 3938.2966309, -4876.4819336, 5006.3808594
1: -1181.6363525, 4660.2524414, -1133.6090088, 4468.8911133, -5650.5273438, 5793.8613281
2: -1201.0739746, 4652.0981445, -1152.3962402, 4461.5258789, -5662.5996094, 5804.4936523
3: -1891.6547852, 4839.5629883, -1815.3112793, 4641.1391602, -6532.7939453, 6654.8740234
4: -1894.5074463, 4631.6870117, -1817.7478027, 4443.6469727, -6338.1533203, 6449.4331055

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_A2_A2_B2_A2_B1

### Relational analysis result of IS_A2_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9766390, upper bound: 5471.9765332
time: 1.00 seconds

## Relational analysis of IS_A2_A2_A2_B2_A2_B2

### Relational analysis result of IS_A2_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9766390, upper bound: 5471.9781644
time: 1.24 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.41 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9782030, upper bound: 5471.9778354
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9785872, upper bound: 5471.9786978
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9782030, upper bound: 5471.9780469
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9785872, upper bound: 5471.9789225
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9782030, upper bound: 5471.9778354
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9785872, upper bound: 5471.9786978
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9782030, upper bound: 5471.9780469
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9785872, upper bound: 5471.9789225
IS_A1_B1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9712459, upper bound: 5471.9758753
IS_A1_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9756296, upper bound: 5471.9775233
IS_A1_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9712459, upper bound: 5471.9781442
IS_A1_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9756296, upper bound: 5471.9787745
IS_A1_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9712459, upper bound: 5471.9758753
IS_A1_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9756296, upper bound: 5471.9775233
IS_A1_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9712459, upper bound: 5471.9784699
IS_A1_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9756296, upper bound: 5471.9787823
IS_A1_B2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9757141, upper bound: 5471.9714709
IS_A1_B2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9784121, upper bound: 5471.9764689
IS_A1_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9787048, upper bound: 5471.9778503
IS_A1_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9787248, upper bound: 5471.9780861
IS_A1_B2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9757649, upper bound: 5471.9722427
IS_A1_B2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9784116, upper bound: 5471.9766914
IS_A1_B2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9774618, upper bound: 5471.9762770
IS_A1_B2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9787227, upper bound: 5471.9782161
IS_A1_B2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9706856, upper bound: 5471.9742252
IS_A1_B2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9748663, upper bound: 5471.9751947
IS_A1_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9706856, upper bound: 5471.9757989
IS_A1_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9748663, upper bound: 5471.9762790
IS_A1_B2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9706856, upper bound: 5471.9754863
IS_A1_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9748662, upper bound: 5471.9768671
IS_A1_B2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9706856, upper bound: 5471.9781152
IS_A1_B2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9748663, upper bound: 5471.9782142
IS_A2_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9784943, upper bound: 5471.9785306
IS_A2_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9784943, upper bound: 5471.9785306
IS_A2_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9784943, upper bound: 5471.9788344
IS_A2_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9784943, upper bound: 5471.9788344
IS_A2_A1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9784794, upper bound: 5471.9773639
IS_A2_A1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9784794, upper bound: 5471.9776668
IS_A2_A1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9786270, upper bound: 5471.9777512
IS_A2_A1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9786366, upper bound: 5471.9780475
IS_A2_A1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9722133, upper bound: 5471.9757766
IS_A2_A1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9765331, upper bound: 5471.9782903
IS_A2_A1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9760855, upper bound: 5471.9773924
IS_A2_A1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9779820, upper bound: 5471.9786670
IS_A2_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9764389, upper bound: 5471.9762880
IS_A2_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9764389, upper bound: 5471.9778351
IS_A2_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9765280, upper bound: 5471.9762889
IS_A2_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9765280, upper bound: 5471.9779530
IS_A2_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9786828, upper bound: 5471.9786828
IS_A2_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9786828, upper bound: 5471.9786828
IS_A2_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9786828, upper bound: 5471.9789233
IS_A2_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9786828, upper bound: 5471.9789233
IS_A2_A2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9775231, upper bound: 5471.9759738
IS_A2_A2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9775229, upper bound: 5471.9779276
IS_A2_A2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9775231, upper bound: 5471.9763305
IS_A2_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9775229, upper bound: 5471.9782085
IS_A2_A2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9721684, upper bound: 5471.9759038
IS_A2_A2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9766199, upper bound: 5471.9784678
IS_A2_A2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9762531, upper bound: 5471.9775214
IS_A2_A2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9781696, upper bound: 5471.9787809
IS_A2_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9765279, upper bound: 5471.9765288
IS_A2_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9765279, upper bound: 5471.9780700
IS_A2_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9766390, upper bound: 5471.9765332
IS_A2_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 3, lower bound: -5471.9766390, upper bound: 5471.9781644

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -772.6435547, 3367.8562012, -772.0126343, 3365.1630859, -4137.8056641, 4139.8686523
1: -973.6626587, 3822.6748047, -972.8847656, 3819.8317871, -4793.4946289, 4795.5595703
2: -989.5352173, 3816.7854004, -988.7546387, 3813.7792969, -4803.3139648, 4805.5400391
3: -1556.5802002, 3972.3762207, -1555.4195557, 3969.4514160, -5526.0312500, 5527.7954102
4: -1560.1916504, 3804.6279297, -1558.9331055, 3801.8972168, -5362.0888672, 5363.5595703

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9769876, upper bound: 5471.9753313
time: 1.06 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9764153, upper bound: 5471.9752414
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -770.4461060, 3358.1274414, -798.3671875, 3482.7805176, -4253.2255859, 4156.4936523
1: -970.9212646, 3811.7045898, -1005.6895142, 3953.2158203, -4924.1362305, 4817.3940430
2: -986.7334595, 3805.8222656, -1021.9967041, 3946.6210938, -4933.3544922, 4827.8188477
3: -1552.1987305, 3961.0219727, -1607.9089355, 4107.5556641, -5659.7543945, 5568.9301758
4: -1555.7185059, 3793.8640137, -1612.3873291, 3932.0578613, -5487.7758789, 5406.2504883

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784251, upper bound: 5471.9785083
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785864, upper bound: 5471.9786978
time: 1.18 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -804.3394775, 3501.3076172, -772.0126343, 3365.1630859, -4169.5019531, 4273.3203125
1: -1012.7216187, 3974.2565918, -972.8847656, 3819.8317871, -4832.5532227, 4947.1396484
2: -1029.6697998, 3967.0075684, -988.7546387, 3813.7792969, -4843.4487305, 4955.7622070
3: -1619.1402588, 4130.0576172, -1555.4195557, 3969.4514160, -5588.5917969, 5685.4755859
4: -1623.4382324, 3954.1323242, -1558.9331055, 3801.8972168, -5425.3349609, 5513.0649414

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9769876, upper bound: 5471.9756324
time: 1.05 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9764153, upper bound: 5471.9755533
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -802.1918945, 3491.7697754, -798.3671875, 3482.7805176, -4284.9711914, 4290.1367188
1: -1010.0460205, 3963.5021973, -1005.6895142, 3953.2158203, -4963.2617188, 4969.1918945
2: -1026.9311523, 3956.2639160, -1021.9967041, 3946.6210938, -4973.5522461, 4978.2607422
3: -1614.8638916, 4118.9179688, -1607.9089355, 4107.5556641, -5722.4194336, 5726.8261719
4: -1619.0638428, 3943.5939941, -1612.3873291, 3932.0578613, -5551.1215820, 5555.9799805

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784251, upper bound: 5471.9787233
time: 1.06 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785864, upper bound: 5471.9789225
time: 1.12 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -772.6435547, 3367.8562012, -803.7313232, 3498.7001953, -4271.3422852, 4171.5874023
1: -973.6626587, 3822.6748047, -1011.9758301, 3971.5051270, -4945.1674805, 4834.6503906
2: -989.5352173, 3816.7854004, -1028.9141846, 3964.1013184, -4953.6362305, 4845.6997070
3: -1556.5802002, 3972.3762207, -1618.0295410, 4127.2036133, -5683.7836914, 5590.4057617
4: -1560.1916504, 3804.6279297, -1622.2098389, 3951.5124512, -5511.7041016, 5426.8364258

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9769243, upper bound: 5471.9767634
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9767359, upper bound: 5471.9752571
time: 1.12 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -770.4461060, 3358.1274414, -830.6480713, 3618.7084961, -4389.1547852, 4188.7739258
1: -970.9212646, 3811.7045898, -1045.4875488, 4107.6401367, -5078.5600586, 4857.1923828
2: -986.7334595, 3805.8222656, -1062.9035645, 4099.7221680, -5086.4541016, 4868.7246094
3: -1552.1987305, 3961.0219727, -1671.6210938, 4268.0844727, -5820.2832031, 5632.6430664
4: -1555.7185059, 3793.8640137, -1676.8348389, 4084.4472656, -5640.1650391, 5470.6982422

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9773961, upper bound: 5471.9779413
time: 1.32 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9773309, upper bound: 5471.9773669
time: 1.06 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -804.3394775, 3501.3076172, -803.7313232, 3498.7001953, -4303.0390625, 4305.0390625
1: -1012.7216187, 3974.2565918, -1011.9758301, 3971.5051270, -4984.2265625, 4986.2314453
2: -1029.6697998, 3967.0075684, -1028.9141846, 3964.1013184, -4993.7709961, 4995.9218750
3: -1619.1402588, 4130.0576172, -1618.0295410, 4127.2036133, -5746.3437500, 5748.0864258
4: -1623.4382324, 3954.1323242, -1622.2098389, 3951.5124512, -5574.9501953, 5576.3417969

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9770110, upper bound: 5471.9756729
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9764755, upper bound: 5471.9755869
time: 1.07 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -802.1918945, 3491.7697754, -830.6480713, 3618.7084961, -4420.9003906, 4322.4179688
1: -1010.0460205, 3963.5021973, -1045.4875488, 4107.6401367, -5117.6860352, 5008.9897461
2: -1026.9311523, 3956.2639160, -1062.9035645, 4099.7221680, -5126.6518555, 5019.1665039
3: -1614.8638916, 4118.9179688, -1671.6210938, 4268.0844727, -5882.9482422, 5790.5390625
4: -1619.0638428, 3943.5939941, -1676.8348389, 4084.4472656, -5703.5112305, 5620.4272461

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784625, upper bound: 5471.9787233
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9786192, upper bound: 5471.9789225
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -836.8504028, 3664.7092285, -779.3359375, 3397.9924316, -4234.8417969, 4444.0449219
1: -1054.7117920, 4159.2958984, -981.8247681, 3856.3466797, -4911.0581055, 5141.1206055
2: -1071.7451172, 4152.1977539, -998.0844116, 3850.5534668, -4922.2978516, 5150.2822266
3: -1688.7973633, 4319.2319336, -1569.5802002, 4007.0463867, -5695.8422852, 5888.8120117
4: -1690.6484375, 4136.1201172, -1574.0927734, 3837.4660645, -5528.1132812, 5710.2128906

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B1_A1_A1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9712060, upper bound: 5471.9758753
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9685113, upper bound: 5471.9739188
time: 0.99 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9683826, upper bound: 5471.9730463
time: 1.11 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -832.1094971, 3641.4064941, -784.8479004, 3422.4487305, -4254.5581055, 4426.2543945
1: -1048.9846191, 4132.3852539, -988.7542114, 3884.2207031, -4933.2045898, 5121.1391602
2: -1065.8403320, 4126.3354492, -1005.1139526, 3878.1459961, -4943.9858398, 5131.4482422
3: -1679.5003662, 4292.2343750, -1580.8433838, 4035.9633789, -5715.4638672, 5873.0776367
4: -1681.1772461, 4110.7348633, -1585.3398438, 3864.9162598, -5546.0922852, 5696.0742188

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B1_A1_A2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9755959, upper bound: 5471.9775233
time: 1.14 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A1_A2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9746240, upper bound: 5471.9756651
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_A2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9735288, upper bound: 5471.9756232
time: 1.12 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -880.9808960, 3856.3366699, -779.3359375, 3397.9924316, -4278.9731445, 4635.6728516
1: -1109.4549561, 4377.3452148, -981.8247681, 3856.3466797, -4965.8017578, 5359.1699219
2: -1127.7741699, 4368.0615234, -998.0844116, 3850.5534668, -4978.3271484, 5366.1450195
3: -1776.6966553, 4544.9506836, -1569.5802002, 4007.0463867, -5783.7412109, 6114.5307617
4: -1778.6827393, 4350.6953125, -1574.0927734, 3837.4660645, -5616.1489258, 5924.7866211

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B1_A2_A1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9752061, upper bound: 5471.9781441
time: 1.06 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9729825, upper bound: 5471.9770671
time: 1.36 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9729078, upper bound: 5471.9764084
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -876.1057739, 3831.8208008, -784.8479004, 3422.4487305, -4298.5546875, 4616.6684570
1: -1103.7222900, 4348.3916016, -988.7542114, 3884.2207031, -4987.9423828, 5337.1450195
2: -1121.8182373, 4341.4082031, -1005.1139526, 3878.1459961, -4999.9643555, 5346.5219727
3: -1767.2618408, 4516.3657227, -1580.8433838, 4035.9633789, -5803.2250977, 6097.2089844
4: -1768.7603760, 4325.0527344, -1585.3398438, 3864.9162598, -5633.6757812, 5910.3920898

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B1_A2_A2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9777104, upper bound: 5471.9787745
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_A2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9777278, upper bound: 5471.9787745
time: 1.13 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -836.8504028, 3664.7092285, -810.8324585, 3530.5410156, -4367.3906250, 4475.5415039
1: -1054.7117920, 4159.2958984, -1020.6170044, 4006.9543457, -5061.6660156, 5179.9116211
2: -1071.7451172, 4152.1977539, -1037.9627686, 3999.7968750, -5071.5415039, 5190.1606445
3: -1688.7973633, 4319.2319336, -1631.7019043, 4163.7602539, -5852.5561523, 5950.9326172
4: -1690.6484375, 4136.1201172, -1636.9373779, 3985.9645996, -5676.6123047, 5773.0576172

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9686213, upper bound: 5471.9739188
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9685543, upper bound: 5471.9730463
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -832.1094971, 3641.4064941, -816.5479736, 3555.8286133, -4387.9370117, 4457.9541016
1: -1048.9846191, 4132.3852539, -1027.8198242, 4035.7226562, -5084.7070312, 5160.2050781
2: -1065.8403320, 4126.3354492, -1045.2440186, 4028.3251953, -5094.1635742, 5171.5791016
3: -1679.5003662, 4292.2343750, -1643.4187012, 4193.6411133, -5873.1411133, 5935.6523438
4: -1681.1772461, 4110.7348633, -1648.5854492, 4014.3920898, -5695.5683594, 5759.3203125

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A1_A2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9758978, upper bound: 5471.9775233
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9739224, upper bound: 5471.9763765
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9738863, upper bound: 5471.9756232
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -880.9808960, 3856.3366699, -810.8324585, 3530.5410156, -4411.5219727, 4667.1689453
1: -1109.4549561, 4377.3452148, -1020.6170044, 4006.9543457, -5116.4091797, 5397.9614258
2: -1127.7741699, 4368.0615234, -1037.9627686, 3999.7968750, -5127.5703125, 5406.0234375
3: -1776.6966553, 4544.9506836, -1631.7019043, 4163.7602539, -5940.4550781, 6176.6518555
4: -1778.6827393, 4350.6953125, -1636.9373779, 3985.9645996, -5764.6474609, 5987.6323242

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A2_A1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9761873, upper bound: 5471.9784699
time: 1.19 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_A1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9761962, upper bound: 5471.9784699
time: 1.30 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -876.1057739, 3831.8208008, -816.5479736, 3555.8286133, -4431.9331055, 4648.3686523
1: -1103.7222900, 4348.3916016, -1027.8198242, 4035.7226562, -5139.4448242, 5376.2109375
2: -1121.8182373, 4341.4082031, -1045.2440186, 4028.3251953, -5150.1425781, 5386.6523438
3: -1767.2618408, 4516.3657227, -1643.4187012, 4193.6411133, -5960.9028320, 6159.7836914
4: -1768.7603760, 4325.0527344, -1648.5854492, 4014.3920898, -5783.1518555, 5973.6381836

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 38

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A2_A2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9779203, upper bound: 5471.9787823
time: 1.35 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_A2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9779374, upper bound: 5471.9787823
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -795.7411499, 3463.0168457, -836.8504028, 3664.7092285, -4460.4501953, 4299.8666992
1: -1002.0167847, 3930.7236328, -1054.7117920, 4159.2958984, -5161.3125000, 4985.4345703
2: -1018.7371826, 3923.9929199, -1071.7451172, 4152.1977539, -5170.9345703, 4995.7368164
3: -1601.7163086, 4085.0373535, -1688.7973633, 4319.2319336, -5920.9482422, 5773.8344727
4: -1605.9156494, 3911.6159668, -1690.6484375, 4136.1201172, -5742.0356445, 5602.2622070

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_B1_B1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757141, upper bound: 5471.9712632
time: 1.33 seconds

## Relational analysis of IS_A1_B2_A1_B1_B1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757141, upper bound: 5471.9714709
time: 1.28 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -801.3651123, 3487.6647949, -880.9808960, 3856.3366699, -4657.7016602, 4368.6455078
1: -1008.9841309, 3958.6933594, -1109.4549561, 4377.3452148, -5386.3291016, 5068.1484375
2: -1025.9112549, 3951.7614746, -1127.7741699, 4368.0615234, -5393.9716797, 5079.5351562
3: -1612.8825684, 4113.9370117, -1776.6966553, 4544.9506836, -6157.8330078, 5890.6308594
4: -1617.2679443, 3939.0065918, -1778.6827393, 4350.6953125, -5967.9619141, 5717.6894531

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_B1_B1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780536, upper bound: 5471.9753922
time: 1.31 seconds

## Relational analysis of IS_A1_B2_A1_B1_B1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780536, upper bound: 5471.9753922
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -772.3026123, 3366.3408203, -875.0755615, 3828.4499512, -4600.7519531, 4241.4165039
1: -973.2321777, 3820.9572754, -1102.4945068, 4344.6767578, -5317.9091797, 4923.4516602
2: -989.1015625, 3815.0720215, -1120.5183105, 4337.5561523, -5326.6572266, 4935.5893555
3: -1555.8886719, 3970.5974121, -1765.5860596, 4512.5156250, -6068.4038086, 5736.1826172
4: -1559.5047607, 3802.9272461, -1766.9288330, 4321.3334961, -5880.8383789, 5569.8559570

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_B1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9774578, upper bound: 5471.9756347
time: 1.05 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9774578, upper bound: 5471.9756347
time: 1.10 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -803.9967651, 3499.7844238, -880.9591675, 3853.2949219, -4657.2915039, 4380.7431641
1: -1012.2904663, 3972.5288086, -1109.8387451, 4372.8066406, -5385.0971680, 5082.3666992
2: -1029.2342529, 3965.2883301, -1128.0290527, 4365.7114258, -5394.9458008, 5093.3164062
3: -1618.4460449, 4128.2661133, -1777.1168213, 4541.6992188, -6160.1445312, 5905.3828125
4: -1622.7464600, 3952.4279785, -1778.6373291, 4349.2329102, -5971.9790039, 5731.0644531

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_B1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9774578, upper bound: 5471.9759406
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9774578, upper bound: 5471.9780861
time: 1.26 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -795.7411499, 3463.0168457, -891.4138184, 3905.6472168, -4701.3881836, 4354.4306641
1: -1002.0167847, 3930.7236328, -1123.1735840, 4432.9287109, -5434.9453125, 5053.8974609
2: -1018.7371826, 3923.9929199, -1141.3878174, 4424.8149414, -5443.5512695, 5065.3808594
3: -1601.7163086, 4085.0373535, -1797.9030762, 4602.9570312, -6204.6733398, 5882.9404297
4: -1605.9156494, 3911.6159668, -1801.0560303, 4404.8403320, -6010.7548828, 5712.6704102

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_B2_B1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757649, upper bound: 5471.9720284
time: 1.11 seconds

## Relational analysis of IS_A1_B2_A1_B2_B1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9757649, upper bound: 5471.9722427
time: 1.09 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -801.3651123, 3487.6647949, -935.8339233, 4098.3437500, -4899.7089844, 4423.4985352
1: -1008.9841309, 3958.6933594, -1178.2315674, 4652.1391602, -5661.1230469, 5136.9248047
2: -1025.9112549, 3951.7614746, -1197.8010254, 4641.8608398, -5667.7709961, 5149.5625000
3: -1612.8825684, 4113.9370117, -1886.2252197, 4829.9155273, -6442.7978516, 6000.1601562
4: -1617.2679443, 3939.0065918, -1889.7298584, 4620.4165039, -6237.6840820, 5828.7363281

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_B2_B1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780612, upper bound: 5471.9757615
time: 1.16 seconds

## Relational analysis of IS_A1_B2_A1_B2_B1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780612, upper bound: 5471.9766914
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -801.7256470, 3489.4672852, -886.9016113, 3883.7211914, -4685.4467773, 4376.3676758
1: -1009.5512085, 3960.8161621, -1117.7502441, 4407.6523438, -5417.2036133, 5078.5659180
2: -1026.3580322, 3953.8190918, -1135.7454834, 4400.5468750, -5426.9047852, 5089.5629883
3: -1613.9632568, 4116.2915039, -1789.1848145, 4577.5278320, -6191.4907227, 5905.4765625
4: -1618.1038818, 3941.3312988, -1791.9608154, 4381.1083984, -5999.2124023, 5733.2919922

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_B2_B2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9774618, upper bound: 5471.9759872
time: 1.17 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9774618, upper bound: 5471.9762770
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -807.1054077, 3513.0661621, -931.1712036, 4075.0515137, -4882.1567383, 4444.2373047
1: -1016.2174683, 3987.5864258, -1172.8034668, 4624.6962891, -5640.9135742, 5160.3896484
2: -1033.2229004, 3980.4111328, -1192.0935059, 4616.7319336, -5649.9541016, 5172.5048828
3: -1624.6468506, 4143.9545898, -1877.3706055, 4802.7070312, -6427.3540039, 6021.3251953
4: -1628.9680176, 3967.5500488, -1880.1920166, 4596.5097656, -6225.4760742, 5847.7412109

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_B2_B2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9787038, upper bound: 5471.9779713
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9787038, upper bound: 5471.9782161
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -836.8504028, 3664.7092285, -843.5886230, 3692.1770020, -4529.0258789, 4508.2973633
1: -1054.7117920, 4159.2958984, -1063.1622314, 4189.5029297, -5244.2148438, 5222.4580078
2: -1071.7451172, 4152.1977539, -1080.4676514, 4183.4882812, -5255.2324219, 5232.6655273
3: -1688.7973633, 4319.2319336, -1702.0583496, 4351.2919922, -6040.0888672, 6021.2900391
4: -1690.6484375, 4136.1201172, -1704.6807861, 4166.6875000, -5857.3349609, 5840.8007812

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B1_A1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9697161, upper bound: 5471.9704699
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9697161, upper bound: 5471.9742252
time: 1.35 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.49 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9769876, upper bound: 5471.9753313
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9764153, upper bound: 5471.9752414
IS_A1_B1_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9784251, upper bound: 5471.9785083
IS_A1_B1_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9785864, upper bound: 5471.9786978
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9769876, upper bound: 5471.9756324
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9764153, upper bound: 5471.9755533
IS_A1_B1_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9784251, upper bound: 5471.9787233
IS_A1_B1_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9785864, upper bound: 5471.9789225
IS_A1_B1_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9769243, upper bound: 5471.9767634
IS_A1_B1_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9767359, upper bound: 5471.9752571
IS_A1_B1_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9773961, upper bound: 5471.9779413
IS_A1_B1_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9773309, upper bound: 5471.9773669
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9770110, upper bound: 5471.9756729
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9764755, upper bound: 5471.9755869
IS_A1_B1_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9784625, upper bound: 5471.9787233
IS_A1_B1_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9786192, upper bound: 5471.9789225
IS_A1_B1_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9685113, upper bound: 5471.9739188
IS_A1_B1_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9683826, upper bound: 5471.9730463
IS_A1_B1_A2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9746240, upper bound: 5471.9756651
IS_A1_B1_A2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9735288, upper bound: 5471.9756232
IS_A1_B1_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9729825, upper bound: 5471.9770671
IS_A1_B1_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9729078, upper bound: 5471.9764084
IS_A1_B1_A2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9777104, upper bound: 5471.9787745
IS_A1_B1_A2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9777278, upper bound: 5471.9787745
IS_A1_B1_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9686213, upper bound: 5471.9739188
IS_A1_B1_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9685543, upper bound: 5471.9730463
IS_A1_B1_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9739224, upper bound: 5471.9763765
IS_A1_B1_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9738863, upper bound: 5471.9756232
IS_A1_B1_A2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9761873, upper bound: 5471.9784699
IS_A1_B1_A2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9761962, upper bound: 5471.9784699
IS_A1_B1_A2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9779203, upper bound: 5471.9787823
IS_A1_B1_A2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9779374, upper bound: 5471.9787823
IS_A1_B2_A1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9757141, upper bound: 5471.9712632
IS_A1_B2_A1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9757141, upper bound: 5471.9714709
IS_A1_B2_A1_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9780536, upper bound: 5471.9753922
IS_A1_B2_A1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9780536, upper bound: 5471.9753922
IS_A1_B2_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9774578, upper bound: 5471.9756347
IS_A1_B2_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9774578, upper bound: 5471.9756347
IS_A1_B2_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9774578, upper bound: 5471.9759406
IS_A1_B2_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9774578, upper bound: 5471.9780861
IS_A1_B2_A1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9757649, upper bound: 5471.9720284
IS_A1_B2_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9757649, upper bound: 5471.9722427
IS_A1_B2_A1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9780612, upper bound: 5471.9757615
IS_A1_B2_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9780612, upper bound: 5471.9766914
IS_A1_B2_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9774618, upper bound: 5471.9759872
IS_A1_B2_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9774618, upper bound: 5471.9762770
IS_A1_B2_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9787038, upper bound: 5471.9779713
IS_A1_B2_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9787038, upper bound: 5471.9782161
IS_A1_B2_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9697161, upper bound: 5471.9704699
IS_A1_B2_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.49
Output dim: 3, lower bound: -5471.9697161, upper bound: 5471.9742252
IS_A1_B2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -5471.9748663, upper bound: 5471.9751947
IS_A1_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -5471.9706856, upper bound: 5471.9757989
IS_A1_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -5471.9748663, upper bound: 5471.9762790
IS_A1_B2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -5471.9706856, upper bound: 5471.9754863
IS_A1_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -5471.9748662, upper bound: 5471.9768671
IS_A1_B2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -5471.9706856, upper bound: 5471.9781152
IS_A1_B2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -5471.9748663, upper bound: 5471.9782142
IS_A2_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -5471.9784943, upper bound: 5471.9785306
IS_A2_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -5471.9784943, upper bound: 5471.9785306
IS_A2_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -5471.9784943, upper bound: 5471.9788344
IS_A2_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -5471.9784943, upper bound: 5471.9788344
IS_A2_A1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -5471.9784794, upper bound: 5471.9773639
IS_A2_A1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -5471.9784794, upper bound: 5471.9776668
IS_A2_A1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -5471.9786270, upper bound: 5471.9777512
IS_A2_A1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -5471.9786366, upper bound: 5471.9780475
IS_A2_A1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -5471.9722133, upper bound: 5471.9757766
IS_A2_A1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -5471.9765331, upper bound: 5471.9782903
IS_A2_A1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -5471.9760855, upper bound: 5471.9773924
IS_A2_A1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -5471.9779820, upper bound: 5471.9786670
IS_A2_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -5471.9764389, upper bound: 5471.9762880
IS_A2_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -5471.9764389, upper bound: 5471.9778351
IS_A2_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -5471.9765280, upper bound: 5471.9762889
IS_A2_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -5471.9765280, upper bound: 5471.9779530
IS_A2_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -5471.9786828, upper bound: 5471.9786828
IS_A2_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -5471.9786828, upper bound: 5471.9786828
IS_A2_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -5471.9786828, upper bound: 5471.9789233
IS_A2_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -5471.9786828, upper bound: 5471.9789233
IS_A2_A2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -5471.9775231, upper bound: 5471.9759738
IS_A2_A2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -5471.9775229, upper bound: 5471.9779276
IS_A2_A2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -5471.9775231, upper bound: 5471.9763305
IS_A2_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -5471.9775229, upper bound: 5471.9782085
IS_A2_A2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -5471.9721684, upper bound: 5471.9759038
IS_A2_A2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -5471.9766199, upper bound: 5471.9784678
IS_A2_A2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -5471.9762531, upper bound: 5471.9775214
IS_A2_A2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -5471.9781696, upper bound: 5471.9787809
IS_A2_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -5471.9765279, upper bound: 5471.9765288
IS_A2_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -5471.9765279, upper bound: 5471.9780700
IS_A2_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -5471.9766390, upper bound: 5471.9765332
IS_A2_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.49
Output dim: 3, lower bound: -5471.9766390, upper bound: 5471.9781644
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=6162.86279296875
rel_dist={3: [-5471.979020840545, 5471.979020840547]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784205, upper bound: 5471.9785249
time: 0.98 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785080, upper bound: 5471.9785080
time: 1.09 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.25 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.25
Output dim: 3, lower bound: -5471.9784205, upper bound: 5471.9785249
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.25
Output dim: 3, lower bound: -5471.9785080, upper bound: 5471.9785080

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -850.7169800, 3697.8376465, -863.1370239, 3753.3532715, -4604.0693359, 4560.9746094
1: -1071.5225830, 4196.7158203, -1086.8886719, 4259.3261719, -5330.8486328, 5283.6035156
2: -1089.2393799, 4190.8291016, -1105.0872803, 4253.2548828, -5342.4931641, 5295.9160156
3: -1711.6445312, 4361.7553711, -1736.3594971, 4426.5043945, -6138.1489258, 6098.1137695
4: -1716.2598877, 4177.3193359, -1741.8470459, 4238.6694336, -5954.9291992, 5919.1665039

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784205, upper bound: 5471.9784205
time: 0.98 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784205, upper bound: 5471.9785080
time: 0.93 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -904.5838623, 3936.6628418, -862.3739014, 3749.9584961, -4654.5410156, 4799.0366211
1: -1139.0826416, 4467.9472656, -1085.9312744, 4255.4711914, -5394.5537109, 5553.8784180
2: -1157.8500977, 4461.1088867, -1104.1119385, 4249.4379883, -5407.2880859, 5565.2207031
3: -1819.3218994, 4642.7978516, -1734.8115234, 4422.5102539, -6241.8320312, 6377.6088867
4: -1825.1307373, 4443.3935547, -1740.2800293, 4234.8916016, -6060.0224609, 6183.6738281

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783046, upper bound: 5471.9782838
time: 1.02 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785002, upper bound: 5471.9785002
time: 1.09 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.00 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.00
Output dim: 3, lower bound: -5471.9784205, upper bound: 5471.9784205
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.00
Output dim: 3, lower bound: -5471.9784205, upper bound: 5471.9785080
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 4.00
Output dim: 3, lower bound: -5471.9783046, upper bound: 5471.9782838
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 4.00
Output dim: 3, lower bound: -5471.9785002, upper bound: 5471.9785002

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -850.7169800, 3697.8376465, -850.7169800, 3697.8376465, -4548.5546875, 4548.5541992
1: -1071.5225830, 4196.7158203, -1071.5225830, 4196.7158203, -5268.2373047, 5268.2373047
2: -1089.2393799, 4190.8291016, -1089.2393799, 4190.8291016, -5280.0678711, 5280.0678711
3: -1711.6445312, 4361.7553711, -1711.6445312, 4361.7553711, -6073.3999023, 6073.3999023
4: -1716.2598877, 4177.3193359, -1716.2598877, 4177.3193359, -5893.5791016, 5893.5791016

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781100, upper bound: 5471.9776649
time: 1.17 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9776739, upper bound: 5471.9776799
time: 1.08 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -850.7169800, 3697.8376465, -904.5838623, 3936.6628418, -4787.3789062, 4602.4213867
1: -1071.5225830, 4196.7158203, -1139.0826416, 4467.9472656, -5539.4687500, 5335.7978516
2: -1089.2393799, 4190.8291016, -1157.8500977, 4461.1088867, -5550.3476562, 5348.6791992
3: -1711.6445312, 4361.7553711, -1819.3218994, 4642.7978516, -6354.4423828, 6181.0771484
4: -1716.2598877, 4177.3193359, -1825.1307373, 4443.3935547, -6159.6533203, 6002.4501953

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783267, upper bound: 5471.9785249
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783809, upper bound: 5471.9785249
time: 0.96 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -877.9072876, 3814.6838379, -860.6748047, 3742.1323242, -4620.0395508, 4675.3583984
1: -1105.8774414, 4330.1918945, -1083.8144531, 4246.6079102, -5352.4853516, 5414.0048828
2: -1124.1602783, 4324.1318359, -1101.9650879, 4240.6430664, -5364.8032227, 5426.0961914
3: -1766.0141602, 4500.7124023, -1731.3924561, 4413.3593750, -6179.3720703, 6232.1049805
4: -1771.0197754, 4310.4692383, -1736.8165283, 4226.3330078, -5997.3525391, 6047.2856445

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783046, upper bound: 5471.9781624
time: 1.01 seconds

## Relational analysis of IS_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_A1_A1

### Relational analysis result of IS_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9779181, upper bound: 5471.9775029
time: 1.19 seconds

## Relational analysis of IS_A2_A1_A2

### Relational analysis result of IS_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9776149, upper bound: 5471.9775029
time: 1.14 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -904.4071655, 3935.9562988, -862.3739014, 3749.9584961, -4654.3647461, 4798.3300781
1: -1138.8594971, 4467.1440430, -1085.9312744, 4255.4711914, -5394.3305664, 5553.0751953
2: -1157.6224365, 4460.3022461, -1104.1119385, 4249.4379883, -5407.0600586, 5564.4140625
3: -1818.9764404, 4641.9545898, -1734.8115234, 4422.5102539, -6241.4858398, 6376.7651367
4: -1824.7785645, 4442.5830078, -1740.2800293, 4234.8916016, -6059.6699219, 6182.8632812

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785002, upper bound: 5471.9784117
time: 1.19 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9785002, upper bound: 5471.9784204
time: 1.47 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.59 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.59
Output dim: 3, lower bound: -5471.9781100, upper bound: 5471.9776649
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.59
Output dim: 3, lower bound: -5471.9776739, upper bound: 5471.9776799
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.59
Output dim: 3, lower bound: -5471.9783267, upper bound: 5471.9785249
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.59
Output dim: 3, lower bound: -5471.9783809, upper bound: 5471.9785249
IS_A2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 4.59
Output dim: 3, lower bound: -5471.9779181, upper bound: 5471.9775029
IS_A2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 4.59
Output dim: 3, lower bound: -5471.9776149, upper bound: 5471.9775029
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 4.59
Output dim: 3, lower bound: -5471.9785002, upper bound: 5471.9784117
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 4.59
Output dim: 3, lower bound: -5471.9785002, upper bound: 5471.9784204

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -809.6653442, 3524.1467285, -827.0213623, 3597.4448242, -4407.1098633, 4351.1674805
1: -1019.4607544, 4000.1494141, -1041.4405518, 4083.1301270, -5102.5908203, 5041.5898438
2: -1036.5018311, 3992.9978027, -1058.8109131, 4076.4580078, -5112.9599609, 5051.8085938
3: -1629.8031006, 4157.0307617, -1664.3463135, 4243.4902344, -5873.2929688, 5821.3769531
4: -1634.1029053, 3980.1218262, -1668.8803711, 4063.3017578, -5697.4042969, 5649.0009766

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9776613, upper bound: 5471.9776612
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9776613, upper bound: 5471.9776612
time: 1.24 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -888.3382568, 3885.9843750, -825.1414795, 3594.8134766, -4483.1518555, 4711.1259766
1: -1119.1064453, 4409.9409180, -1039.2912598, 4079.3291016, -5198.4345703, 5449.2314453
2: -1137.4340820, 4402.6269531, -1056.5384521, 4073.6696777, -5211.1020508, 5459.1650391
3: -1792.0124512, 4580.1577148, -1661.5618896, 4238.7558594, -6030.7685547, 6241.7197266
4: -1793.5903320, 4385.8295898, -1665.5273438, 4059.1708984, -5852.7602539, 6051.3569336

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9776613, upper bound: 5471.9776799
time: 0.99 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9776613, upper bound: 5471.9776799
time: 1.22 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -837.7370605, 3639.8962402, -904.5838623, 3936.6628418, -4774.3989258, 4544.4794922
1: -1055.4593506, 4131.5219727, -1139.0826416, 4467.9472656, -5523.4062500, 5270.6040039
2: -1072.6903076, 4125.6699219, -1157.8500977, 4461.1088867, -5533.7993164, 5283.5200195
3: -1685.8868408, 4294.3632812, -1819.3218994, 4642.7978516, -6328.6845703, 6113.6850586
4: -1689.5224609, 4113.5068359, -1825.1307373, 4443.3935547, -6132.9160156, 5938.6376953

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9779541, upper bound: 5471.9783235
time: 1.15 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783150, upper bound: 5471.9785245
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -864.8975220, 3761.0805664, -897.4774170, 3905.4843750, -4770.3808594, 4658.5581055
1: -1089.3088379, 4269.1293945, -1130.1923828, 4432.7250977, -5522.0336914, 5399.3208008
2: -1106.9920654, 4262.6191406, -1148.7906494, 4425.8666992, -5532.8583984, 5411.4091797
3: -1740.1057129, 4436.8198242, -1805.1331787, 4606.3149414, -6346.4199219, 6241.9526367
4: -1744.6708984, 4247.9169922, -1810.7866211, 4408.6030273, -6153.2739258, 6058.7036133

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9775473, upper bound: 5471.9782086
time: 1.24 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9775690, upper bound: 5471.9778218
time: 1.14 seconds

## BFS IS instance: IS_A2_A1_A1

### Backsubstitution after applying IS history:
0: -836.8653564, 3641.1677246, -837.0651245, 3642.1213379, -4478.9858398, 4478.2324219
1: -1053.7990723, 4133.8930664, -1053.8355713, 4133.4506836, -5187.2485352, 5187.7285156
2: -1071.4710693, 4126.4687500, -1071.6451416, 4126.7104492, -5198.1816406, 5198.1137695
3: -1684.1738281, 4296.1738281, -1684.2625732, 4295.5351562, -5979.7084961, 5980.4365234
4: -1688.9489746, 4113.4960938, -1689.6093750, 4112.7558594, -5801.7045898, 5803.1049805

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_A1_A1_B1

### Relational analysis result of IS_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9776019, upper bound: 5471.9775029
time: 1.18 seconds

## Relational analysis of IS_A2_A1_A1_B2

### Relational analysis result of IS_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9776019, upper bound: 5471.9775029
time: 1.08 seconds

## BFS IS instance: IS_A2_A1_A2

### Backsubstitution after applying IS history:
0: -915.8217163, 4003.0617676, -835.1623535, 3639.3420410, -4555.1630859, 4838.2231445
1: -1153.8229980, 4543.6396484, -1051.6513672, 4129.4697266, -5283.2929688, 5595.2910156
2: -1172.8452148, 4536.1694336, -1069.3382568, 4123.7680664, -5296.6132812, 5605.5078125
3: -1846.8117676, 4719.3300781, -1681.4277344, 4290.6342773, -6137.4458008, 6400.7568359
4: -1848.9246826, 4519.6440430, -1686.2058105, 4108.4487305, -5957.3730469, 6205.8486328

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_A1_A2_B1

### Relational analysis result of IS_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9776019, upper bound: 5471.9775029
time: 1.25 seconds

## Relational analysis of IS_A2_A1_A2_B2

### Relational analysis result of IS_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9776019, upper bound: 5471.9775029
time: 1.32 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -904.4071655, 3935.9562988, -850.7169800, 3697.8376465, -4602.2446289, 4786.6733398
1: -1138.8594971, 4467.1440430, -1071.5225830, 4196.7158203, -5335.5747070, 5538.6660156
2: -1157.6224365, 4460.3022461, -1089.2393799, 4190.8291016, -5348.4516602, 5549.5410156
3: -1818.9764404, 4641.9545898, -1711.6445312, 4361.7553711, -6180.7309570, 6353.5991211
4: -1824.7785645, 4442.5830078, -1716.2598877, 4177.3193359, -6002.0976562, 6158.8427734

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_A2_B1_A1

### Relational analysis result of IS_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782085, upper bound: 5471.9776356
time: 1.10 seconds

## Relational analysis of IS_A2_A2_B1_A2

### Relational analysis result of IS_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9777954, upper bound: 5471.9776634
time: 1.04 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -904.4071655, 3935.9562988, -904.5838623, 3936.6628418, -4841.0693359, 4840.5400391
1: -1138.8594971, 4467.1440430, -1139.0826416, 4467.9472656, -5606.8066406, 5606.2265625
2: -1157.6224365, 4460.3022461, -1157.8500977, 4461.1088867, -5618.7314453, 5618.1523438
3: -1818.9764404, 4641.9545898, -1819.3218994, 4642.7978516, -6461.7739258, 6461.2763672
4: -1824.7785645, 4442.5830078, -1825.1307373, 4443.3935547, -6268.1718750, 6267.7138672

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782085, upper bound: 5471.9776843
time: 1.07 seconds

## Relational analysis of IS_A2_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9777954, upper bound: 5471.9776993
time: 1.00 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.50 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.50
Output dim: 3, lower bound: -5471.9776613, upper bound: 5471.9776612
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.50
Output dim: 3, lower bound: -5471.9776613, upper bound: 5471.9776612
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.50
Output dim: 3, lower bound: -5471.9776613, upper bound: 5471.9776799
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.50
Output dim: 3, lower bound: -5471.9776613, upper bound: 5471.9776799
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.50
Output dim: 3, lower bound: -5471.9779541, upper bound: 5471.9783235
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.50
Output dim: 3, lower bound: -5471.9783150, upper bound: 5471.9785245
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.50
Output dim: 3, lower bound: -5471.9775473, upper bound: 5471.9782086
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.50
Output dim: 3, lower bound: -5471.9775690, upper bound: 5471.9778218
IS_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.50
Output dim: 3, lower bound: -5471.9776019, upper bound: 5471.9775029
IS_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.50
Output dim: 3, lower bound: -5471.9776019, upper bound: 5471.9775029
IS_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.50
Output dim: 3, lower bound: -5471.9776019, upper bound: 5471.9775029
IS_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.50
Output dim: 3, lower bound: -5471.9776019, upper bound: 5471.9775029
IS_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.50
Output dim: 3, lower bound: -5471.9782085, upper bound: 5471.9776356
IS_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.50
Output dim: 3, lower bound: -5471.9777954, upper bound: 5471.9776634
IS_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.50
Output dim: 3, lower bound: -5471.9782085, upper bound: 5471.9776843
IS_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.50
Output dim: 3, lower bound: -5471.9777954, upper bound: 5471.9776993

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -809.6653442, 3524.1467285, -809.6653442, 3524.1467285, -4333.8115234, 4333.8115234
1: -1019.4607544, 4000.1494141, -1019.4607544, 4000.1494141, -5019.6103516, 5019.6103516
2: -1036.5018311, 3992.9978027, -1036.5018311, 3992.9978027, -5029.4995117, 5029.4995117
3: -1629.8031006, 4157.0307617, -1629.8031006, 4157.0307617, -5786.8339844, 5786.8339844
4: -1634.1029053, 3980.1218262, -1634.1029053, 3980.1218262, -5614.2246094, 5614.2246094

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9763388, upper bound: 5471.9753187
time: 1.08 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2

### Relational analysis result of IS_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9763388, upper bound: 5471.9776649
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -809.6653442, 3524.1467285, -888.3382568, 3885.9843750, -4695.6494141, 4412.4843750
1: -1019.4607544, 4000.1494141, -1119.1064453, 4409.9409180, -5429.4018555, 5119.2558594
2: -1036.5018311, 3992.9978027, -1137.4340820, 4402.6269531, -5439.1289062, 5130.4311523
3: -1629.8031006, 4157.0307617, -1792.0124512, 4580.1577148, -6209.9609375, 5949.0429688
4: -1634.1029053, 3980.1218262, -1793.5903320, 4385.8295898, -6019.9326172, 5773.7114258

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780799, upper bound: 5471.9775487
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780798, upper bound: 5471.9775652
time: 1.06 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -888.3382568, 3885.9843750, -809.6653442, 3524.1467285, -4412.4843750, 4695.6489258
1: -1119.1064453, 4409.9409180, -1019.4607544, 4000.1494141, -5119.2558594, 5429.4018555
2: -1137.4340820, 4402.6269531, -1036.5018311, 3992.9978027, -5130.4311523, 5439.1289062
3: -1792.0124512, 4580.1577148, -1629.8031006, 4157.0307617, -5949.0429688, 6209.9609375
4: -1793.5903320, 4385.8295898, -1634.1029053, 3980.1218262, -5773.7114258, 6019.9326172

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9775443, upper bound: 5471.9775781
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9775622, upper bound: 5471.9775761
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -888.3382568, 3885.9843750, -888.3382568, 3885.9843750, -4774.3222656, 4774.3222656
1: -1119.1064453, 4409.9409180, -1119.1064453, 4409.9409180, -5529.0463867, 5529.0463867
2: -1137.4340820, 4402.6269531, -1137.4340820, 4402.6269531, -5540.0605469, 5540.0605469
3: -1792.0124512, 4580.1577148, -1792.0124512, 4580.1577148, -6372.1699219, 6372.1699219
4: -1793.5903320, 4385.8295898, -1793.5903320, 4385.8295898, -6179.4199219, 6179.4199219

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9761238, upper bound: 5471.9775249
time: 1.05 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9776497, upper bound: 5471.9776608
time: 1.46 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -836.0307617, 3632.0693359, -877.9072876, 3814.6838379, -4650.7143555, 4509.9765625
1: -1053.3291016, 4122.6455078, -1105.8774414, 4330.1918945, -5383.5200195, 5228.5229492
2: -1070.5418701, 4116.8696289, -1124.1602783, 4324.1318359, -5394.6733398, 5241.0297852
3: -1682.4554443, 4285.1982422, -1766.0141602, 4500.7124023, -6183.1679688, 6051.2109375
4: -1686.0588379, 4104.9423828, -1771.0197754, 4310.4692383, -5996.5283203, 5875.9619141

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9772157, upper bound: 5471.9779178
time: 1.10 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2

### Relational analysis result of IS_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9772157, upper bound: 5471.9776165
time: 1.14 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -837.7370605, 3639.8962402, -904.4071655, 3935.9562988, -4773.6933594, 4544.3032227
1: -1055.4593506, 4131.5219727, -1138.8594971, 4467.1440430, -5522.6035156, 5270.3808594
2: -1072.6903076, 4125.6699219, -1157.6224365, 4460.3022461, -5532.9926758, 5283.2924805
3: -1685.8868408, 4294.3632812, -1818.9764404, 4641.9545898, -6327.8413086, 6113.3393555
4: -1689.5224609, 4113.5068359, -1824.7785645, 4442.5830078, -6132.1054688, 5938.2851562

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9775234, upper bound: 5471.9782086
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2

### Relational analysis result of IS_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9775502, upper bound: 5471.9778034
time: 1.30 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -841.2774658, 3660.9313965, -856.0224609, 3730.1049805, -4571.3823242, 4516.9536133
1: -1059.3120117, 4155.8041992, -1077.6149902, 4234.2841797, -5293.5952148, 5233.4184570
2: -1076.6606445, 4148.5771484, -1095.5959473, 4226.2021484, -5302.8627930, 5244.1728516
3: -1692.8930664, 4318.7441406, -1722.4429932, 4399.5771484, -6092.4692383, 6041.1865234
4: -1697.4156494, 4134.1860352, -1727.8616943, 4209.6464844, -5907.0605469, 5862.0463867

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9775473, upper bound: 5471.9778058
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9775473, upper bound: 5471.9778218
time: 1.13 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -839.3829346, 3658.3437500, -936.1826172, 4097.7124023, -4937.0952148, 4594.5244141
1: -1057.1749268, 4152.1245117, -1179.1457520, 4650.6342773, -5707.8085938, 5331.2695312
2: -1074.3546143, 4145.7685547, -1198.4747314, 4642.2773438, -5716.6318359, 5344.2431641
3: -1690.1533203, 4314.1127930, -1887.7368164, 4829.5288086, -6519.6821289, 6201.8496094
4: -1693.9913330, 4130.0546875, -1890.4287109, 4622.0434570, -6316.0346680, 6020.4829102

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9775690, upper bound: 5471.9778058
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9775690, upper bound: 5471.9778218
time: 1.47 seconds

## BFS IS instance: IS_A2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -836.8653564, 3641.1677246, -819.8026733, 3569.2319336, -4406.0966797, 4460.9702148
1: -1053.7990723, 4133.8930664, -1031.9752197, 4050.9165039, -5104.7148438, 5165.8676758
2: -1071.4710693, 4126.4687500, -1049.4504395, 4043.7187500, -5115.1899414, 5175.9189453
3: -1684.1738281, 4296.1738281, -1649.9099121, 4209.5576172, -5893.7309570, 5946.0839844
4: -1688.9489746, 4113.4960938, -1655.0186768, 4030.0415039, -5718.9902344, 5768.5146484

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_A1_A1_B1_B1

### Relational analysis result of IS_A2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9779178, upper bound: 5471.9773713
time: 1.08 seconds

## Relational analysis of IS_A2_A1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_A1_B1_B1

### Relational analysis result of IS_A2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9778611, upper bound: 5471.9771483
time: 1.02 seconds

## Relational analysis of IS_A2_A1_A1_B1_B2

### Relational analysis result of IS_A2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9779181, upper bound: 5471.9775029
time: 1.20 seconds

## BFS IS instance: IS_A2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -836.8653564, 3641.1677246, -898.1380615, 3929.5000000, -4766.3642578, 4539.3056641
1: -1053.7990723, 4133.8930664, -1131.2014160, 4458.9179688, -5512.7167969, 5265.0947266
2: -1071.4710693, 4126.4687500, -1149.9605713, 4451.6337891, -5523.1049805, 5276.4287109
3: -1684.1738281, 4296.1738281, -1811.4306641, 4630.8364258, -6315.0102539, 6107.6044922
4: -1688.9489746, 4113.4960938, -1813.8247070, 4433.9946289, -6122.9428711, 5927.3203125

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_A1_A1_B2_B1

### Relational analysis result of IS_A2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9779178, upper bound: 5471.9773713
time: 0.96 seconds

## Relational analysis of IS_A2_A1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_A1_B2_B1

### Relational analysis result of IS_A2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9778611, upper bound: 5471.9771483
time: 1.08 seconds

## Relational analysis of IS_A2_A1_A1_B2_B2

### Relational analysis result of IS_A2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9779181, upper bound: 5471.9775029
time: 1.02 seconds

## BFS IS instance: IS_A2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -915.8217163, 4003.0617676, -819.8026733, 3569.2319336, -4485.0532227, 4822.8642578
1: -1153.8229980, 4543.6396484, -1031.9752197, 4050.9165039, -5204.7387695, 5575.6147461
2: -1172.8452148, 4536.1694336, -1049.4504395, 4043.7187500, -5216.5639648, 5585.6201172
3: -1846.8117676, 4719.3300781, -1649.9099121, 4209.5576172, -6056.3686523, 6369.2402344
4: -1848.9246826, 4519.6440430, -1655.0186768, 4030.0415039, -5878.9658203, 6174.6625977

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_A1_A2_B1_B1

### Relational analysis result of IS_A2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9776019, upper bound: 5471.9773712
time: 1.07 seconds

## Relational analysis of IS_A2_A1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_A1_A2_B1_A1

### Relational analysis result of IS_A2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9761620, upper bound: 5471.9773360
time: 1.20 seconds

## Relational analysis of IS_A2_A1_A2_B1_A2

### Relational analysis result of IS_A2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9775935, upper bound: 5471.9774909
time: 1.32 seconds

## BFS IS instance: IS_A2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -915.8217163, 4003.0617676, -898.1380615, 3929.5000000, -4845.3203125, 4901.1997070
1: -1153.8229980, 4543.6396484, -1131.2014160, 4458.9179688, -5612.7407227, 5674.8408203
2: -1172.8452148, 4536.1694336, -1149.9605713, 4451.6337891, -5624.4790039, 5686.1293945
3: -1846.8117676, 4719.3300781, -1811.4306641, 4630.8364258, -6477.6474609, 6530.7602539
4: -1848.9246826, 4519.6440430, -1813.8247070, 4433.9946289, -6282.9184570, 6333.4682617

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_A1_A2_B2_B1

### Relational analysis result of IS_A2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9776019, upper bound: 5471.9773712
time: 1.36 seconds

## Relational analysis of IS_A2_A1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_A1_A2_B2_A1

### Relational analysis result of IS_A2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9761620, upper bound: 5471.9773360
time: 1.12 seconds

## Relational analysis of IS_A2_A1_A2_B2_A2

### Relational analysis result of IS_A2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9775935, upper bound: 5471.9774909
time: 1.19 seconds

## BFS IS instance: IS_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -862.8826294, 3760.1909180, -827.0213623, 3597.4448242, -4460.3266602, 4587.2114258
1: -1086.2114258, 4268.2900391, -1041.4405518, 4083.1301270, -5169.3417969, 5309.7304688
2: -1104.3425293, 4260.2314453, -1058.8109131, 4076.4580078, -5180.8002930, 5319.0424805
3: -1736.1597900, 4434.8232422, -1664.3463135, 4243.4902344, -5979.6499023, 6099.1694336
4: -1741.6884766, 4243.2480469, -1668.8803711, 4063.3017578, -5804.9897461, 5912.1274414

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_A2_B1_A1_B1

### Relational analysis result of IS_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9776575, upper bound: 5471.9776356
time: 1.13 seconds

## Relational analysis of IS_A2_A2_B1_A1_B2

### Relational analysis result of IS_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9777897, upper bound: 5471.9776356
time: 0.99 seconds

## BFS IS instance: IS_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -942.9987793, 4127.6245117, -825.1414795, 3594.8134766, -4537.8125000, 4952.7661133
1: -1187.6744385, 4684.4360352, -1039.2912598, 4079.3291016, -5267.0034180, 5723.7275391
2: -1207.1914062, 4676.1494141, -1056.5384521, 4073.6696777, -5280.8598633, 5732.6870117
3: -1901.3382568, 4864.5913086, -1661.5618896, 4238.7558594, -6140.0942383, 6526.1523438
4: -1904.2214355, 4655.4824219, -1665.5273438, 4059.1708984, -5963.3906250, 6321.0087891

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_A2_B1_A2_B1

### Relational analysis result of IS_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9777897, upper bound: 5471.9776634
time: 1.06 seconds

## Relational analysis of IS_A2_A2_B1_A2_B2

### Relational analysis result of IS_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9777897, upper bound: 5471.9776634
time: 1.16 seconds

## BFS IS instance: IS_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -862.8826294, 3760.1909180, -880.6910400, 3835.3427734, -4698.2255859, 4640.8808594
1: -1086.2114258, 4268.2900391, -1108.7797852, 4353.3833008, -5439.5947266, 5377.0698242
2: -1104.3425293, 4260.2314453, -1127.1999512, 4345.7998047, -5450.1416016, 5387.4316406
3: -1736.1597900, 4434.8232422, -1771.6373291, 4523.5039062, -6259.6635742, 6206.4604492
4: -1741.6884766, 4243.2480469, -1777.3596191, 4328.5073242, -6070.1958008, 6020.6074219

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_A2_B2_A1_B1

### Relational analysis result of IS_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9777796, upper bound: 5471.9776843
time: 1.03 seconds

## Relational analysis of IS_A2_A2_B2_A1_B2

### Relational analysis result of IS_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9776356, upper bound: 5471.9776356
time: 1.17 seconds

## BFS IS instance: IS_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -942.9987793, 4127.6245117, -879.5390015, 3836.4189453, -4779.4179688, 5007.1635742
1: -1187.6744385, 4684.4360352, -1107.5396729, 4353.6997070, -5541.3740234, 5791.9750977
2: -1207.1914062, 4676.1494141, -1125.8945312, 4347.0336914, -5554.2250977, 5802.0429688
3: -1901.3382568, 4864.5913086, -1770.3715820, 4522.9169922, -6424.2553711, 6634.9614258
4: -1904.2214355, 4655.4824219, -1775.5136719, 4328.3227539, -6232.5434570, 6430.9946289

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_A2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9776356, upper bound: 5471.9776992
time: 1.13 seconds

## Relational analysis of IS_A2_A2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9777796, upper bound: 5471.9776992
time: 1.02 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.22 seconds
IS_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 3, lower bound: -5471.9763388, upper bound: 5471.9753187
IS_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 3, lower bound: -5471.9763388, upper bound: 5471.9776649
IS_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 3, lower bound: -5471.9780799, upper bound: 5471.9775487
IS_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 3, lower bound: -5471.9780798, upper bound: 5471.9775652
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 3, lower bound: -5471.9775443, upper bound: 5471.9775781
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 3, lower bound: -5471.9775622, upper bound: 5471.9775761
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 3, lower bound: -5471.9761238, upper bound: 5471.9775249
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 3, lower bound: -5471.9776497, upper bound: 5471.9776608
IS_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 3, lower bound: -5471.9772157, upper bound: 5471.9779178
IS_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 3, lower bound: -5471.9772157, upper bound: 5471.9776165
IS_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 3, lower bound: -5471.9775234, upper bound: 5471.9782086
IS_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 3, lower bound: -5471.9775502, upper bound: 5471.9778034
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 3, lower bound: -5471.9775473, upper bound: 5471.9778058
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 3, lower bound: -5471.9775473, upper bound: 5471.9778218
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 3, lower bound: -5471.9775690, upper bound: 5471.9778058
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 3, lower bound: -5471.9775690, upper bound: 5471.9778218
IS_A2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 3, lower bound: -5471.9778611, upper bound: 5471.9771483
IS_A2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 3, lower bound: -5471.9779181, upper bound: 5471.9775029
IS_A2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 3, lower bound: -5471.9778611, upper bound: 5471.9771483
IS_A2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 3, lower bound: -5471.9779181, upper bound: 5471.9775029
IS_A2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 3, lower bound: -5471.9761620, upper bound: 5471.9773360
IS_A2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 3, lower bound: -5471.9775935, upper bound: 5471.9774909
IS_A2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 3, lower bound: -5471.9761620, upper bound: 5471.9773360
IS_A2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 3, lower bound: -5471.9775935, upper bound: 5471.9774909
IS_A2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 3, lower bound: -5471.9776575, upper bound: 5471.9776356
IS_A2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 3, lower bound: -5471.9777897, upper bound: 5471.9776356
IS_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 3, lower bound: -5471.9777897, upper bound: 5471.9776634
IS_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 3, lower bound: -5471.9777897, upper bound: 5471.9776634
IS_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 3, lower bound: -5471.9777796, upper bound: 5471.9776843
IS_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 3, lower bound: -5471.9776356, upper bound: 5471.9776356
IS_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 3, lower bound: -5471.9776356, upper bound: 5471.9776992
IS_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.22
Output dim: 3, lower bound: -5471.9777796, upper bound: 5471.9776992

## BFS IS instance: IS_A1_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -796.3603516, 3466.1137695, -772.6435547, 3367.8562012, -4164.2167969, 4238.7563477
1: -1002.8531494, 3934.3464355, -973.6626587, 3822.6748047, -4825.5278320, 4908.0083008
2: -1019.4930420, 3927.4101562, -989.5352173, 3816.7854004, -4836.2783203, 4916.9448242
3: -1603.2796631, 4088.8737793, -1556.5802002, 3972.3762207, -5575.6557617, 5645.4531250
4: -1607.3051758, 3915.1979980, -1560.1916504, 3804.6279297, -5411.9321289, 5475.3886719

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782365, upper bound: 5471.9782365
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9782365, upper bound: 5471.9784413
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -806.1609497, 3509.0678711, -804.3394775, 3501.3076172, -4307.4687500, 4313.4067383
1: -1015.0212402, 3983.0546875, -1012.7216187, 3974.2565918, -4989.2773438, 4995.7763672
2: -1032.0076904, 3975.8454590, -1029.6697998, 3967.0075684, -4999.0146484, 5005.5151367
3: -1622.7672119, 4139.2246094, -1619.1402588, 4130.0576172, -5752.8247070, 5758.3647461
4: -1627.0859375, 3962.9682617, -1623.4382324, 3954.1323242, -5581.2182617, 5586.4057617

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784413, upper bound: 5471.9782365
time: 1.32 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9784413, upper bound: 5471.9784413
time: 1.19 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -809.6653442, 3524.1467285, -875.5478516, 3828.9350586, -4638.6000977, 4399.6938477
1: -1019.4607544, 4000.1494141, -1103.2596436, 4345.8090820, -5365.2700195, 5103.4091797
2: -1036.5018311, 3992.9978027, -1121.1326904, 4338.4038086, -5374.9052734, 5114.1303711
3: -1629.8031006, 4157.0307617, -1766.6218262, 4513.8168945, -6143.6196289, 5923.6523438
4: -1634.1029053, 3980.1218262, -1767.2308350, 4323.0278320, -5957.1308594, 5747.3510742

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780235, upper bound: 5471.9775487
time: 1.17 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780235, upper bound: 5471.9775487
time: 1.38 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -805.8773193, 3507.4208984, -902.6672363, 3949.5439453, -4755.4213867, 4410.0874023
1: -1014.7398682, 3981.2966309, -1137.0457764, 4482.7773438, -5497.5170898, 5118.3422852
2: -1031.6728516, 3974.1430664, -1155.3479004, 4474.6499023, -5506.3217773, 5129.4912109
3: -1622.2729492, 4137.4990234, -1820.8013916, 4655.5537109, -6277.8266602, 5958.3002930
4: -1626.4038086, 3961.6320801, -1822.2174072, 4456.8876953, -6083.2915039, 5783.8496094

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780235, upper bound: 5471.9775652
time: 1.07 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9780235, upper bound: 5471.9775652
time: 1.21 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -875.5478516, 3828.9350586, -809.6653442, 3524.1467285, -4399.6938477, 4638.6000977
1: -1103.2596436, 4345.8090820, -1019.4607544, 4000.1494141, -5103.4091797, 5365.2700195
2: -1121.1326904, 4338.4038086, -1036.5018311, 3992.9978027, -5114.1303711, 5374.9052734
3: -1766.6218262, 4513.8168945, -1629.8031006, 4157.0307617, -5923.6523438, 6143.6196289
4: -1767.2308350, 4323.0278320, -1634.1029053, 3980.1218262, -5747.3510742, 5957.1308594

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9775487, upper bound: 5471.9780235
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9775487, upper bound: 5471.9780798
time: 1.13 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -902.6672363, 3949.5439453, -805.8773193, 3507.4208984, -4410.0874023, 4755.4208984
1: -1137.0457764, 4482.7773438, -1014.7398682, 3981.2966309, -5118.3422852, 5497.5170898
2: -1155.3479004, 4474.6499023, -1031.6728516, 3974.1430664, -5129.4912109, 5506.3217773
3: -1820.8013916, 4655.5537109, -1622.2729492, 4137.4990234, -5958.3002930, 6277.8266602
4: -1822.2174072, 4456.8876953, -1626.4038086, 3961.6320801, -5783.8496094, 6083.2915039

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9775652, upper bound: 5471.9780235
time: 1.10 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9775652, upper bound: 5471.9780798
time: 1.18 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -888.8275146, 3891.2319336, -876.8561401, 3834.5576172, -4723.3842773, 4768.0869141
1: -1119.3469238, 4417.0312500, -1104.6411133, 4351.5161133, -5470.8632812, 5521.6723633
2: -1137.8276367, 4407.5161133, -1122.8138428, 4344.6049805, -5482.4316406, 5530.3295898
3: -1792.6622314, 4586.1245117, -1768.4082031, 4519.5312500, -6312.1933594, 6354.5327148
4: -1794.6807861, 4389.9921875, -1770.1892090, 4328.0961914, -6122.7763672, 6160.1806641

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9760640, upper bound: 5471.9760810
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9760640, upper bound: 5471.9775249
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -883.2570190, 3863.5400391, -885.9545288, 3875.4357910, -4758.6928711, 4749.4936523
1: -1112.7313232, 4384.4619141, -1116.1154785, 4397.9638672, -5510.6948242, 5500.5771484
2: -1130.9761963, 4377.2978516, -1134.4020996, 4390.7221680, -5521.6982422, 5511.6997070
3: -1781.7916260, 4553.7802734, -1787.2150879, 4567.7612305, -6349.5522461, 6340.9946289
4: -1783.3317871, 4360.7568359, -1788.7733154, 4374.0444336, -6157.3750000, 6149.5302734

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9760891, upper bound: 5471.9760943
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9760891, upper bound: 5471.9776608
time: 1.10 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -812.2967529, 3531.5075684, -836.8653564, 3641.1677246, -4453.4643555, 4368.3720703
1: -1023.1992188, 4008.8864746, -1053.7990723, 4133.8930664, -5157.0917969, 5062.6850586
2: -1040.0646973, 4002.3029785, -1071.4710693, 4126.4687500, -5166.5332031, 5073.7739258
3: -1635.0805664, 4166.7309570, -1684.1738281, 4296.1738281, -5931.2543945, 5850.9042969
4: -1638.6037598, 3990.7260742, -1688.9489746, 4113.4960938, -5752.0996094, 5679.6743164

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A1_B1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9772157, upper bound: 5471.9776039
time: 1.27 seconds

## Relational analysis of IS_A1_B2_A1_B1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9772157, upper bound: 5471.9776165
time: 1.28 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -810.5380249, 3529.4514160, -915.8217163, 4003.0617676, -4813.5996094, 4445.2719727
1: -1021.2047119, 4005.7314453, -1153.8229980, 4543.6396484, -5564.8442383, 5159.5541992
2: -1037.9488525, 4000.1623535, -1172.8452148, 4536.1694336, -5574.1176758, 5173.0078125
3: -1632.5502930, 4162.6699219, -1846.8117676, 4719.3300781, -6351.8803711, 6009.4809570
4: -1635.5017090, 3987.2458496, -1848.9246826, 4519.6440430, -6155.1445312, 5836.1699219

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A1_B1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9772157, upper bound: 5471.9776039
time: 1.11 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9772157, upper bound: 5471.9776165
time: 1.09 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -813.9560547, 3539.1149902, -862.8826294, 3760.1909180, -4574.1455078, 4401.9975586
1: -1025.2694092, 4017.5124512, -1086.2114258, 4268.2900391, -5293.5595703, 5103.7236328
2: -1042.1524658, 4010.8562012, -1104.3425293, 4260.2314453, -5302.3833008, 5115.1977539
3: -1638.4145508, 4175.6391602, -1736.1597900, 4434.8232422, -6073.2373047, 5911.7983398
4: -1641.9724121, 3999.0458984, -1741.6884766, 4243.2480469, -5885.2207031, 5740.7343750

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9772157, upper bound: 5471.9777898
time: 1.23 seconds

## Relational analysis of IS_A1_B2_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9775234, upper bound: 5471.9778034
time: 1.19 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -812.2873535, 3537.4499512, -942.9987793, 4127.6245117, -4939.9121094, 4480.4487305
1: -1023.3903198, 4014.8034668, -1187.6744385, 4684.4360352, -5707.8261719, 5202.4780273
2: -1040.1523438, 4009.1586914, -1207.1914062, 4676.1494141, -5716.3017578, 5216.3500977
3: -1636.0668945, 4172.0385742, -1901.3382568, 4864.5913086, -6500.6582031, 6073.3769531
4: -1639.0506592, 3996.0043945, -1904.2214355, 4655.4824219, -6294.5317383, 5900.2250977

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9775502, upper bound: 5471.9777898
time: 1.11 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9775502, upper bound: 5471.9778034
time: 1.07 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -823.9565430, 3587.8649902, -856.0224609, 3730.1049805, -4554.0605469, 4443.8867188
1: -1037.3558350, 4073.0419922, -1077.6149902, 4234.2841797, -5271.6401367, 5150.6572266
2: -1054.4107666, 4065.3466797, -1095.5959473, 4226.2021484, -5280.6127930, 5160.9423828
3: -1658.3800049, 4232.4960938, -1722.4429932, 4399.5771484, -6057.9555664, 5954.9384766
4: -1662.7570801, 4051.1474609, -1727.8616943, 4209.6464844, -5872.4033203, 5779.0078125

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9752954, upper bound: 5471.9766461
time: 1.09 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9775473, upper bound: 5471.9782073
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -902.6672363, 3949.5439453, -856.0224609, 3730.1049805, -4632.7700195, 4805.5654297
1: -1137.0457764, 4482.7773438, -1077.6149902, 4234.2841797, -5371.3300781, 5560.3925781
2: -1155.3479004, 4474.6499023, -1095.5959473, 4226.2021484, -5381.5498047, 5570.2451172
3: -1820.8013916, 4655.5537109, -1722.4429932, 4399.5771484, -6220.3774414, 6377.9965820
4: -1822.2174072, 4456.8876953, -1727.8616943, 4209.6464844, -6031.8637695, 6184.7485352

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9772334, upper bound: 5471.9779178
time: 1.27 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9775384, upper bound: 5471.9782086
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -823.9565430, 3587.8649902, -936.1826172, 4097.7124023, -4921.6689453, 4524.0473633
1: -1037.3558350, 4073.0419922, -1179.1457520, 4650.6342773, -5687.9902344, 5252.1875000
2: -1054.4107666, 4065.3466797, -1198.4747314, 4642.2773438, -5696.6879883, 5263.8212891
3: -1658.3800049, 4232.4960938, -1887.7368164, 4829.5288086, -6487.9086914, 6120.2329102
4: -1662.7570801, 4051.1474609, -1890.4287109, 4622.0434570, -6284.8007812, 5941.5756836

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9746815, upper bound: 5471.9748736
time: 1.25 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9775473, upper bound: 5471.9778058
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -902.6672363, 3949.5439453, -936.1826172, 4097.7124023, -5000.3789062, 4885.7255859
1: -1137.0457764, 4482.7773438, -1179.1457520, 4650.6342773, -5787.6801758, 5661.9228516
2: -1155.3479004, 4474.6499023, -1198.4747314, 4642.2773438, -5797.6250000, 5673.1240234
3: -1820.8013916, 4655.5537109, -1887.7368164, 4829.5288086, -6650.3300781, 6543.2905273
4: -1822.2174072, 4456.8876953, -1890.4287109, 4622.0434570, -6444.2607422, 6347.3159180

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9746815, upper bound: 5471.9748736
time: 1.22 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9775473, upper bound: 5471.9778218
time: 0.97 seconds

## BFS IS instance: IS_A2_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -836.8653564, 3641.1677246, -806.6154175, 3510.4594727, -4347.3237305, 4447.7832031
1: -1053.7990723, 4133.8930664, -1015.6621094, 3984.8288574, -5038.6274414, 5149.5551758
2: -1071.4710693, 4126.4687500, -1032.6516113, 3977.6379395, -5049.1088867, 5159.1201172
3: -1684.1738281, 4296.1738281, -1623.7723389, 4141.1835938, -5825.3574219, 5919.9453125
4: -1688.9489746, 4113.4960938, -1627.8950195, 3965.3195801, -5654.2670898, 5741.3901367

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_A1_A1_B1_B1_B1

### Relational analysis result of IS_A2_A1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9779072, upper bound: 5471.9772341
time: 1.11 seconds

## Relational analysis of IS_A2_A1_A1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_A1_A1_B1_B1_A1

### Relational analysis result of IS_A2_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9779035, upper bound: 5471.9770986
time: 1.06 seconds

## Relational analysis of IS_A2_A1_A1_B1_B1_A2

### Relational analysis result of IS_A2_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9779035, upper bound: 5471.9772341
time: 1.25 seconds

## BFS IS instance: IS_A2_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -829.7031250, 3609.9089355, -831.5369873, 3622.0749512, -4451.7783203, 4441.4458008
1: -1044.8238525, 4098.5268555, -1046.6787109, 4111.3754883, -5156.1982422, 5145.2055664
2: -1062.3242188, 4091.0922852, -1064.1354980, 4103.6972656, -5166.0214844, 5155.2275391
3: -1669.8737793, 4259.4868164, -1673.4105225, 4272.0810547, -5941.9550781, 5932.8974609
4: -1674.4301758, 4078.5585938, -1678.5401611, 4088.6840820, -5763.1137695, 5757.0986328

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_A1_A1_B1_B2_B1

### Relational analysis result of IS_A2_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9783046, upper bound: 5471.9780521
time: 1.43 seconds

## Relational analysis of IS_A2_A1_A1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_A1_A1_B1_B2_B1

### Relational analysis result of IS_A2_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781227, upper bound: 5471.9781227
time: 1.13 seconds

## Relational analysis of IS_A2_A1_A1_B1_B2_B2

### Relational analysis result of IS_A2_A1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9781227, upper bound: 5471.9782838
time: 1.11 seconds

## BFS IS instance: IS_A2_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -836.8653564, 3641.1677246, -885.2891846, 3872.2949219, -4709.1596680, 4526.4565430
1: -1053.7990723, 4133.8930664, -1115.3026123, 4394.6137695, -5448.4130859, 5249.1948242
2: -1071.4710693, 4126.4687500, -1133.5904541, 4387.2426758, -5458.7133789, 5260.0590820
3: -1684.1738281, 4296.1738281, -1785.9605713, 4564.3105469, -6248.4843750, 6082.1342773
4: -1688.9489746, 4113.4960938, -1787.3776855, 4370.9897461, -6059.9379883, 5900.8730469

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_A1_A1_B2_B1_B1

### Relational analysis result of IS_A2_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9778611, upper bound: 5471.9771483
time: 1.06 seconds

## Relational analysis of IS_A2_A1_A1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_A1_A1_B2_B1_A1

### Relational analysis result of IS_A2_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9775740, upper bound: 5471.9763391
time: 1.02 seconds

## Relational analysis of IS_A2_A1_A1_B2_B1_A2

### Relational analysis result of IS_A2_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9778578, upper bound: 5471.9771483
time: 1.12 seconds

## BFS IS instance: IS_A2_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -829.7031250, 3609.9089355, -909.8592529, 3982.0041504, -4811.7055664, 4519.7670898
1: -1044.8238525, 4098.5268555, -1145.9063721, 4519.1645508, -5563.9873047, 5244.4331055
2: -1062.3242188, 4091.0922852, -1164.5767822, 4511.0849609, -5573.4091797, 5255.6684570
3: -1669.8737793, 4259.4868164, -1835.0718994, 4693.1035156, -6362.9775391, 6094.5585938
4: -1674.4301758, 4078.5585938, -1837.2218018, 4492.4897461, -6166.9199219, 5915.7802734

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_A1_A1_B2_B2_B1

### Relational analysis result of IS_A2_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9779178, upper bound: 5471.9772334
time: 1.43 seconds

## Relational analysis of IS_A2_A1_A1_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_A1_A1_B2_B2_A1

### Relational analysis result of IS_A2_A1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9775740, upper bound: 5471.9765787
time: 1.26 seconds

## Relational analysis of IS_A2_A1_A1_B2_B2_A2

### Relational analysis result of IS_A2_A1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9779159, upper bound: 5471.9775029
time: 1.15 seconds

## BFS IS instance: IS_A2_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -916.0964355, 4006.4870605, -809.7889404, 3524.8793945, -4440.9750977, 4816.2758789
1: -1153.8082275, 4548.4433594, -1019.3549805, 4000.4741211, -5154.2807617, 5567.7978516
2: -1172.9799805, 4539.1308594, -1036.6943359, 3993.6872559, -5166.6665039, 5575.8251953
3: -1846.9881592, 4723.2812500, -1629.3895264, 4157.1577148, -6004.1459961, 6352.6708984
4: -1849.4436035, 4522.1342773, -1634.6058350, 3980.2104492, -5829.6542969, 6156.7397461

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_A1_A2_B1_A1_B1

### Relational analysis result of IS_A2_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9761620, upper bound: 5471.9773311
time: 1.13 seconds

## Relational analysis of IS_A2_A1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_A2_B1_A1_B1

### Relational analysis result of IS_A2_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9759953, upper bound: 5471.9767990
time: 1.03 seconds

## Relational analysis of IS_A2_A1_A2_B1_A1_B2

### Relational analysis result of IS_A2_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9761620, upper bound: 5471.9774194
time: 1.41 seconds

## BFS IS instance: IS_A2_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -911.4382324, 3983.6218262, -817.3574219, 3558.3820801, -4469.8203125, 4800.9794922
1: -1148.3144531, 4521.5424805, -1028.8999023, 4038.6147461, -5186.9291992, 5550.4423828
2: -1167.2752686, 4514.2280273, -1046.3446045, 4031.4731445, -5198.7485352, 5560.5722656
3: -1837.9632568, 4696.4526367, -1644.9619141, 4196.8076172, -6034.7695312, 6341.4145508
4: -1840.0678711, 4497.9003906, -1650.0888672, 4017.8972168, -5857.9648438, 6147.9892578

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_A1_A2_B1_A2_B1

### Relational analysis result of IS_A2_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9775945, upper bound: 5471.9779215
time: 1.14 seconds

## Relational analysis of IS_A2_A1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_A1_A2_B1_A2_A1

### Relational analysis result of IS_A2_A1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9772138, upper bound: 5471.9773824
time: 1.06 seconds

## Relational analysis of IS_A2_A1_A2_B1_A2_A2

### Relational analysis result of IS_A2_A1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9769398, upper bound: 5471.9773280
time: 1.09 seconds

## BFS IS instance: IS_A2_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -916.0964355, 4006.4870605, -886.6794434, 3878.1589355, -4794.2553711, 4893.1655273
1: -1153.8082275, 4548.4433594, -1116.7624512, 4400.5898438, -5554.3969727, 5665.2055664
2: -1172.9799805, 4539.1308594, -1135.3726807, 4393.7060547, -5566.6855469, 5674.5034180
3: -1846.9881592, 4723.2812500, -1787.8703613, 4570.3134766, -6417.3017578, 6511.1513672
4: -1849.4436035, 4522.1342773, -1790.4656982, 4376.3608398, -6225.8046875, 6312.5996094

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_A1_A2_B2_A1_B1

### Relational analysis result of IS_A2_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9760711, upper bound: 5471.9759136
time: 1.19 seconds

## Relational analysis of IS_A2_A1_A2_B2_A1_B2

### Relational analysis result of IS_A2_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -5471.9760711, upper bound: 5471.9773360
time: 1.07 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.42 seconds
IS_A1_B1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -5471.9782365, upper bound: 5471.9782365
IS_A1_B1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -5471.9782365, upper bound: 5471.9784413
IS_A1_B1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -5471.9784413, upper bound: 5471.9782365
IS_A1_B1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -5471.9784413, upper bound: 5471.9784413
IS_A1_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -5471.9780235, upper bound: 5471.9775487
IS_A1_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -5471.9780235, upper bound: 5471.9775487
IS_A1_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -5471.9780235, upper bound: 5471.9775652
IS_A1_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -5471.9780235, upper bound: 5471.9775652
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -5471.9775487, upper bound: 5471.9780235
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -5471.9775487, upper bound: 5471.9780798
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -5471.9775652, upper bound: 5471.9780235
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -5471.9775652, upper bound: 5471.9780798
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -5471.9760640, upper bound: 5471.9760810
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -5471.9760640, upper bound: 5471.9775249
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -5471.9760891, upper bound: 5471.9760943
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -5471.9760891, upper bound: 5471.9776608
IS_A1_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -5471.9772157, upper bound: 5471.9776039
IS_A1_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -5471.9772157, upper bound: 5471.9776165
IS_A1_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -5471.9772157, upper bound: 5471.9776039
IS_A1_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -5471.9772157, upper bound: 5471.9776165
IS_A1_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -5471.9772157, upper bound: 5471.9777898
IS_A1_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -5471.9775234, upper bound: 5471.9778034
IS_A1_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -5471.9775502, upper bound: 5471.9777898
IS_A1_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -5471.9775502, upper bound: 5471.9778034
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -5471.9752954, upper bound: 5471.9766461
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -5471.9775473, upper bound: 5471.9782073
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -5471.9772334, upper bound: 5471.9779178
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -5471.9775384, upper bound: 5471.9782086
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -5471.9746815, upper bound: 5471.9748736
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -5471.9775473, upper bound: 5471.9778058
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -5471.9746815, upper bound: 5471.9748736
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -5471.9775473, upper bound: 5471.9778218
IS_A2_A1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -5471.9779035, upper bound: 5471.9770986
IS_A2_A1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -5471.9779035, upper bound: 5471.9772341
IS_A2_A1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -5471.9781227, upper bound: 5471.9781227
IS_A2_A1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -5471.9781227, upper bound: 5471.9782838
IS_A2_A1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -5471.9775740, upper bound: 5471.9763391
IS_A2_A1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -5471.9778578, upper bound: 5471.9771483
IS_A2_A1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -5471.9775740, upper bound: 5471.9765787
IS_A2_A1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -5471.9779159, upper bound: 5471.9775029
IS_A2_A1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -5471.9759953, upper bound: 5471.9767990
IS_A2_A1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -5471.9761620, upper bound: 5471.9774194
IS_A2_A1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -5471.9772138, upper bound: 5471.9773824
IS_A2_A1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -5471.9769398, upper bound: 5471.9773280
IS_A2_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -5471.9760711, upper bound: 5471.9759136
IS_A2_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.42
Output dim: 3, lower bound: -5471.9760711, upper bound: 5471.9773360
IS_A2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 3, lower bound: -5471.9775935, upper bound: 5471.9774909
IS_A2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 3, lower bound: -5471.9776575, upper bound: 5471.9776356
IS_A2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 3, lower bound: -5471.9777897, upper bound: 5471.9776356
IS_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 3, lower bound: -5471.9777897, upper bound: 5471.9776634
IS_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 3, lower bound: -5471.9777897, upper bound: 5471.9776634
IS_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 3, lower bound: -5471.9777796, upper bound: 5471.9776843
IS_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 3, lower bound: -5471.9776356, upper bound: 5471.9776356
IS_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 3, lower bound: -5471.9776356, upper bound: 5471.9776992
IS_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.42
Output dim: 3, lower bound: -5471.9777796, upper bound: 5471.9776992
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=6162.86279296875
rel_dist={3: [-5471.978598948089, 5471.97859894809]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 1107.50 seconds
