## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_4.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 495.22538199974406


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-226.9094849, 344.3358154, -226.9094849, 344.3358154, -571.2452393, 571.2453003)
1: (-253.4875641, 367.0639648, -253.4875641, 367.0639648, -620.5515137, 620.5515137)
2: (-257.4188232, 361.7965088, -257.4188232, 361.7965088, -619.2153320, 619.2153320)
3: (-309.8564758, 425.3124084, -309.8564758, 425.3124084, -735.1688843, 735.1688843)
4: (-281.1933594, 418.6253052, -281.1933594, 418.6253052, -699.8186646, 699.8186646)

## BASE Result
execution time: IAR + LP analysis = 2.51 + 2.57 = 5.08 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -495.2996525, upper bound: 495.2996525


# Binary Search by BASE starts (time budget: 1194.92 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=571.2453002929688
rel_dist={0: [-495.2995013334902, 495.2995013334903]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=571.2453002929688
rel_dist={0: [-495.2824179308574, 495.28241793085726]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=571.2453002929688
rel_dist={0: [-495.2676133325849, 495.2676133325849]}

## Binary search (step 3) starts
Candidate diff: 0.0625000


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=571.2453002929688
rel_dist={0: [-495.25922920202663, 495.2592292020265]}

## Binary search (step 4) starts
Candidate diff: 0.0312500


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=571.2453002929688
rel_dist={0: [-495.25475054336266, 495.25475054336266]}

## Binary search (step 5) starts
Candidate diff: 0.0156250


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=571.2453002929688
rel_dist={0: [-495.25236467462537, 495.25236467462537]}

## Binary search (step 6) starts
Candidate diff: 0.0078125


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0078125, mid=0.0078125, abs_max=571.2453002929688
rel_dist={0: [-495.2511477260622, 495.2511477260623]}

## Binary search (step 7) starts
Candidate diff: 0.0039062


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0039062, mid=0.0039062, abs_max=571.2453002929688
rel_dist={0: [-495.25052896198576, 495.25052896198576]}

## Binary search (step 8) starts
Candidate diff: 0.0019531


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0019531, mid=0.0019531, abs_max=571.2453002929688
rel_dist={0: [-495.25021539278737, 495.25021539278737]}

## Binary search (step 9) starts
Candidate diff: 0.0009766


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0009766, mid=0.0009766, abs_max=571.2453002929688
rel_dist={0: [-495.2500586082076, 495.2500586082076]}

## Binary search (step 10) starts
Candidate diff: 0.0004883


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0004883, mid=0.0004883, abs_max=571.2453002929688
rel_dist={0: [-495.2499769975135, 495.2499769975134]}

## Binary search (step 11) starts
Candidate diff: 0.0002441


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0002441, mid=0.0002441, abs_max=571.2453002929688
rel_dist={0: [-495.24993524612717, 495.24993524612705]}

## Binary search (step 12) starts
Candidate diff: 0.0001221


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0001221, mid=0.0001221, abs_max=571.2453002929688
rel_dist={0: [-495.2499143704797, 495.2499143704797]}

## Binary search (step 13) starts
Candidate diff: 0.0000610


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000610, mid=0.0000610, abs_max=571.2453002929688
rel_dist={0: [-495.24990393274675, 495.24990393274675]}

## Binary search (step 14) starts
Candidate diff: 0.0000305


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000305, mid=0.0000305, abs_max=571.2453002929688
rel_dist={0: [-495.2498987140587, 495.24989871405865]}

## Binary search (step 15) starts
Candidate diff: 0.0000153


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000153, mid=0.0000153, abs_max=571.2453002929688
rel_dist={0: [-495.2498961106081, 495.2498961061506]}

## Binary search (step 16) starts
Candidate diff: 0.0000076


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000076, mid=0.0000076, abs_max=571.2453002929688
rel_dist={0: [-495.24989480884824, 495.24989480884824]}

## Binary search (step 17) starts
Candidate diff: 0.0000038


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000038, mid=0.0000038, abs_max=571.2453002929688
rel_dist={0: [-495.2498941753107, 495.2498941494455]}

## Binary search (step 18) starts
Candidate diff: 0.0000019


## IAR start
Binary search (step 18): status=Status.UNKNOWN, low=0.0000000, high=0.0000019, mid=0.0000019, abs_max=571.2453002929688
rel_dist={0: [-495.2498938466084, 495.2498938332592]}

## Binary search (step 19) starts
Candidate diff: 0.0000010


## IAR start
Binary search (step 19): status=Status.UNKNOWN, low=0.0000000, high=0.0000010, mid=0.0000010, abs_max=571.2453002929688
rel_dist={0: [-495.24989370687626, 495.24989374739084]}

## Binary Search Result
Binary search time: 104.40 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 1090.52 seconds

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2887365, upper bound: 495.2914396
time: 0.93 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2866167, upper bound: 495.2866167
time: 1.27 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.39 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.39
Output dim: 0, lower bound: -495.2887365, upper bound: 495.2914396
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.39
Output dim: 0, lower bound: -495.2866167, upper bound: 495.2866167

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -217.9161682, 331.5028992, -226.9094849, 344.3358154, -562.2519531, 558.4123535
1: -243.3218842, 353.3270569, -253.4875641, 367.0639648, -610.3858643, 606.8145142
2: -247.2664490, 348.2680054, -257.4188232, 361.7965088, -609.0629883, 605.6867065
3: -297.4977417, 409.3919373, -309.8564758, 425.3124084, -722.8101196, 719.2484131
4: -270.4958191, 402.7216187, -281.1933594, 418.6253052, -689.1210327, 683.9149780

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2866167, upper bound: 495.2866167
time: 1.06 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2866167, upper bound: 495.2866167
time: 1.19 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -225.6279755, 342.1831360, -226.9094849, 344.3358154, -569.9636841, 569.0926514
1: -251.9228973, 365.0004272, -253.4875641, 367.0639648, -618.9868774, 618.4879761
2: -256.0439148, 359.9935608, -257.4188232, 361.7965088, -617.8404541, 617.4122314
3: -307.8818359, 422.5994568, -309.8564758, 425.3124084, -733.1942139, 732.4559326
4: -279.6287537, 416.4534912, -281.1933594, 418.6253052, -698.2540283, 697.6468506

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2866167, upper bound: 495.2866167
time: 1.42 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2866167, upper bound: 495.2866167
time: 1.26 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.97 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.97
Output dim: 0, lower bound: -495.2866167, upper bound: 495.2866167
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.97
Output dim: 0, lower bound: -495.2866167, upper bound: 495.2866167
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.97
Output dim: 0, lower bound: -495.2866167, upper bound: 495.2866167
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.97
Output dim: 0, lower bound: -495.2866167, upper bound: 495.2866167

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -217.9161682, 331.5028992, -217.9161682, 331.5028992, -549.4190674, 549.4190674
1: -243.3218842, 353.3270569, -243.3218842, 353.3270569, -596.6488647, 596.6488037
2: -247.2664490, 348.2680054, -247.2664490, 348.2680054, -595.5344238, 595.5344238
3: -297.4977417, 409.3919373, -297.4977417, 409.3919373, -706.8896484, 706.8896484
4: -270.4958191, 402.7216187, -270.4958191, 402.7216187, -673.2174072, 673.2174072

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2564693, upper bound: 495.2884463
time: 1.78 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2589608, upper bound: 495.2914396
time: 1.16 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -217.9161682, 331.5028992, -225.6279755, 342.1831360, -560.0993042, 557.1308594
1: -243.3218842, 353.3270569, -251.9228973, 365.0004272, -608.3223267, 605.2498779
2: -247.2664490, 348.2680054, -256.0439148, 359.9935608, -607.2600098, 604.3118896
3: -297.4977417, 409.3919373, -307.8818359, 422.5994568, -720.0971680, 717.2738037
4: -270.4958191, 402.7216187, -279.6287537, 416.4534912, -686.9491577, 682.3503418

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2564693, upper bound: 495.2884463
time: 1.40 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2589608, upper bound: 495.2914396
time: 1.85 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -225.6279755, 342.1831360, -217.9161682, 331.5028992, -557.1308594, 560.0993042
1: -251.9228973, 365.0004272, -243.3218842, 353.3270569, -605.2499390, 608.3223267
2: -256.0439148, 359.9935608, -247.2664490, 348.2680054, -604.3118896, 607.2600098
3: -307.8818359, 422.5994568, -297.4977417, 409.3919373, -717.2738037, 720.0971680
4: -279.6287537, 416.4534912, -270.4958191, 402.7216187, -682.3503418, 686.9491577

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2564827, upper bound: 495.2866167
time: 1.44 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2565248, upper bound: 495.2565248
time: 1.11 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -225.6279755, 342.1831360, -225.6279755, 342.1831360, -567.8110962, 567.8110962
1: -251.9228973, 365.0004272, -251.9228973, 365.0004272, -616.9233398, 616.9233398
2: -256.0439148, 359.9935608, -256.0439148, 359.9935608, -616.0373535, 616.0373535
3: -307.8818359, 422.5994568, -307.8818359, 422.5994568, -730.4813232, 730.4812622
4: -279.6287537, 416.4534912, -279.6287537, 416.4534912, -696.0822144, 696.0822144

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2564827, upper bound: 495.2866167
time: 1.01 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2565248, upper bound: 495.2565248
time: 1.20 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.50 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.50
Output dim: 0, lower bound: -495.2564693, upper bound: 495.2884463
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.50
Output dim: 0, lower bound: -495.2589608, upper bound: 495.2914396
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.50
Output dim: 0, lower bound: -495.2564693, upper bound: 495.2884463
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.50
Output dim: 0, lower bound: -495.2589608, upper bound: 495.2914396
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.50
Output dim: 0, lower bound: -495.2564827, upper bound: 495.2866167
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.50
Output dim: 0, lower bound: -495.2565248, upper bound: 495.2565248
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.50
Output dim: 0, lower bound: -495.2564827, upper bound: 495.2866167
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.50
Output dim: 0, lower bound: -495.2565248, upper bound: 495.2565248

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -212.5932922, 323.4444580, -217.9161682, 331.5028992, -544.0961914, 541.3605957
1: -237.4012604, 344.7285461, -243.3218842, 353.3270569, -590.7282715, 588.0504150
2: -241.2593231, 339.8170166, -247.2664490, 348.2680054, -589.5272827, 587.0834961
3: -290.2413635, 399.4304810, -297.4977417, 409.3919373, -699.6333008, 696.9281616
4: -264.0952454, 392.8693237, -270.4958191, 402.7216187, -666.8168945, 663.3651123

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2884660, upper bound: 495.2884660
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2884660, upper bound: 495.2909100
time: 1.12 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -205.4523468, 312.1578369, -217.7147217, 331.1842957, -536.6366577, 529.8725586
1: -229.4598083, 332.8937378, -243.0978088, 352.9918823, -582.4516602, 575.9914551
2: -233.1933289, 328.2497253, -247.0371399, 347.9385071, -581.1318359, 575.2868652
3: -280.7006531, 385.4885559, -297.2263184, 408.9982910, -689.6987305, 682.7147827
4: -255.1674500, 379.6665955, -270.2449951, 402.3468018, -657.5142822, 649.9116211

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2909100, upper bound: 495.2914514
time: 1.99 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2909100, upper bound: 495.2940538
time: 1.80 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -212.5932922, 323.4444580, -225.6279755, 342.1831360, -554.7764282, 549.0723267
1: -237.4012604, 344.7285461, -251.9228973, 365.0004272, -602.4016724, 596.6514282
2: -241.2593231, 339.8170166, -256.0439148, 359.9935608, -601.2527466, 595.8608398
3: -290.2413635, 399.4304810, -307.8818359, 422.5994568, -712.8408203, 707.3123169
4: -264.0952454, 392.8693237, -279.6287537, 416.4534912, -680.5485840, 672.4980469

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2564693, upper bound: 495.2884401
time: 1.44 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2564693, upper bound: 495.2884463
time: 1.22 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -205.4523468, 312.1578369, -225.4316254, 341.8752747, -547.3276367, 537.5894775
1: -229.4598083, 332.8937378, -251.7046204, 364.6795654, -594.1394043, 584.5983887
2: -233.1933289, 328.2497253, -255.8205872, 359.6760864, -592.8693848, 584.0703125
3: -280.7006531, 385.4885559, -307.6151733, 422.2221680, -702.9226685, 693.1037598
4: -255.1674500, 379.6665955, -279.3883057, 416.0845032, -671.2518921, 659.0549316

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2589608, upper bound: 495.2914126
time: 1.63 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2589608, upper bound: 495.2914396
time: 1.74 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -220.2614441, 334.0695190, -217.9161682, 331.5028992, -551.7643433, 551.9857178
1: -245.9608459, 356.2183838, -243.3218842, 353.3270569, -599.2877197, 599.5402832
2: -249.9733276, 351.4279785, -247.2664490, 348.2680054, -598.2413330, 598.6944580
3: -300.6258850, 412.4967651, -297.4977417, 409.3919373, -710.0178223, 709.9944458
4: -273.0295410, 406.6106567, -270.4958191, 402.7216187, -675.7511597, 677.1064453

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2884401, upper bound: 495.2564693
time: 2.06 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2884401, upper bound: 495.2589608
time: 1.41 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -215.4856720, 326.4540405, -217.7147217, 331.1842957, -546.6699829, 544.1687622
1: -240.6076508, 348.3619995, -243.0978088, 352.9918823, -593.5994263, 591.4595947
2: -244.4766083, 343.6737366, -247.0371399, 347.9385071, -592.4151001, 590.7108765
3: -294.2843933, 403.1745911, -297.2263184, 408.9982910, -703.2825317, 700.4008789
4: -267.0398254, 397.5583801, -270.2449951, 402.3468018, -669.3865967, 667.8032837

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2884463, upper bound: 495.2564693
time: 1.25 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2884463, upper bound: 495.2589608
time: 1.22 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -220.2614441, 334.0695190, -225.6279755, 342.1831360, -562.4445190, 559.6974487
1: -245.9608459, 356.2183838, -251.9228973, 365.0004272, -610.9611816, 608.1412964
2: -249.9733276, 351.4279785, -256.0439148, 359.9935608, -609.9668579, 607.4718018
3: -300.6258850, 412.4967651, -307.8818359, 422.5994568, -723.2253418, 720.3786011
4: -273.0295410, 406.6106567, -279.6287537, 416.4534912, -689.4830322, 686.2393799

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2564827, upper bound: 495.2564665
time: 0.93 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2564827, upper bound: 495.2565248
time: 1.17 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -215.4856720, 326.4540405, -225.4316254, 341.8752747, -557.3609619, 551.8856812
1: -240.6076508, 348.3619995, -251.7046204, 364.6795654, -605.2871094, 600.0665894
2: -244.4766083, 343.6737366, -255.8205872, 359.6760864, -604.1526489, 599.4943237
3: -294.2843933, 403.1745911, -307.6151733, 422.2221680, -716.5065308, 710.7897949
4: -267.0398254, 397.5583801, -279.3883057, 416.0845032, -683.1243286, 676.9466553

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2565248, upper bound: 495.2564665
time: 1.25 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2565248, upper bound: 495.2565248
time: 1.13 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.77 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 0, lower bound: -495.2884660, upper bound: 495.2884660
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 0, lower bound: -495.2884660, upper bound: 495.2909100
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 0, lower bound: -495.2909100, upper bound: 495.2914514
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 0, lower bound: -495.2909100, upper bound: 495.2940538
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 0, lower bound: -495.2564693, upper bound: 495.2884401
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 0, lower bound: -495.2564693, upper bound: 495.2884463
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 0, lower bound: -495.2589608, upper bound: 495.2914126
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 0, lower bound: -495.2589608, upper bound: 495.2914396
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 0, lower bound: -495.2884401, upper bound: 495.2564693
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 0, lower bound: -495.2884401, upper bound: 495.2589608
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 0, lower bound: -495.2884463, upper bound: 495.2564693
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 0, lower bound: -495.2884463, upper bound: 495.2589608
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 0, lower bound: -495.2564827, upper bound: 495.2564665
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 0, lower bound: -495.2564827, upper bound: 495.2565248
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 0, lower bound: -495.2565248, upper bound: 495.2564665
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.77
Output dim: 0, lower bound: -495.2565248, upper bound: 495.2565248

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -212.5932922, 323.4444580, -212.5932922, 323.4444580, -536.0377197, 536.0377197
1: -237.4012604, 344.7285461, -237.4012604, 344.7285461, -582.1298218, 582.1298218
2: -241.2593231, 339.8170166, -241.2593231, 339.8170166, -581.0762329, 581.0761719
3: -290.2413635, 399.4304810, -290.2413635, 399.4304810, -689.6718750, 689.6718750
4: -264.0952454, 392.8693237, -264.0952454, 392.8693237, -656.9645386, 656.9645996

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2862479, upper bound: 495.2858725
time: 1.09 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2884660, upper bound: 495.2884660
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -212.5932922, 323.4444580, -205.4523468, 312.1578369, -524.7510986, 528.8967896
1: -237.4012604, 344.7285461, -229.4598083, 332.8937378, -570.2949829, 574.1883545
2: -241.2593231, 339.8170166, -233.1933289, 328.2497253, -569.5089722, 573.0103149
3: -290.2413635, 399.4304810, -280.7006531, 385.4885559, -675.7299194, 680.1310425
4: -264.0952454, 392.8693237, -255.1674500, 379.6665955, -643.7617798, 648.0367432

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2862479, upper bound: 495.2884129
time: 1.11 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2884660, upper bound: 495.2909100
time: 1.07 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -205.4523468, 312.1578369, -212.5932922, 323.4444580, -528.8967896, 524.7510986
1: -229.4598083, 332.8937378, -237.4012604, 344.7285461, -574.1883545, 570.2949829
2: -233.1933289, 328.2497253, -241.2593231, 339.8170166, -573.0103149, 569.5090332
3: -280.7006531, 385.4885559, -290.2413635, 399.4304810, -680.1310425, 675.7299194
4: -255.1674500, 379.6665955, -264.0952454, 392.8693237, -648.0367432, 643.7618408

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2838063, upper bound: 495.2656104
time: 1.42 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2909100, upper bound: 495.2914514
time: 1.05 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -205.4523468, 312.1578369, -205.4523468, 312.1578369, -517.6101685, 517.6101685
1: -229.4598083, 332.8937378, -229.4598083, 332.8937378, -562.3535156, 562.3535156
2: -233.1933289, 328.2497253, -233.1933289, 328.2497253, -561.4430542, 561.4430542
3: -280.7006531, 385.4885559, -280.7006531, 385.4885559, -666.1890869, 666.1890869
4: -255.1674500, 379.6665955, -255.1674500, 379.6665955, -634.8340454, 634.8340454

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2838063, upper bound: 495.2656104
time: 1.04 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2909100, upper bound: 495.2940538
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -212.5932922, 323.4444580, -220.2614441, 334.0695190, -546.6628418, 543.7058105
1: -237.4012604, 344.7285461, -245.9608459, 356.2183838, -593.6196289, 590.6893311
2: -241.2593231, 339.8170166, -249.9733276, 351.4279785, -592.6871338, 589.7903442
3: -290.2413635, 399.4304810, -300.6258850, 412.4967651, -702.7381592, 700.0563965
4: -264.0952454, 392.8693237, -273.0295410, 406.6106567, -670.7059326, 665.8988647

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2511311, upper bound: 495.2858466
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2564693, upper bound: 495.2884401
time: 3.17 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -212.5932922, 323.4444580, -215.4856720, 326.4540405, -539.0473022, 538.9301147
1: -237.4012604, 344.7285461, -240.6076508, 348.3619995, -585.7632446, 585.3361816
2: -241.2593231, 339.8170166, -244.4766083, 343.6737366, -584.9328613, 584.2936401
3: -290.2413635, 399.4304810, -294.2843933, 403.1745911, -693.4159546, 693.7148438
4: -264.0952454, 392.8693237, -267.0398254, 397.5583801, -661.6534424, 659.9091797

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2511311, upper bound: 495.2858527
time: 1.59 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2564693, upper bound: 495.2884463
time: 1.09 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -205.4523468, 312.1578369, -220.2614441, 334.0695190, -539.5218506, 532.4193115
1: -229.4598083, 332.8937378, -245.9608459, 356.2183838, -585.6782227, 578.8544922
2: -233.1933289, 328.2497253, -249.9733276, 351.4279785, -584.6212769, 578.2230225
3: -280.7006531, 385.4885559, -300.6258850, 412.4967651, -693.1973267, 686.1144409
4: -255.1674500, 379.6665955, -273.0295410, 406.6106567, -661.7780762, 652.6961670

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2443541, upper bound: 495.2655846
time: 1.18 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2589608, upper bound: 495.2914126
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -205.4523468, 312.1578369, -215.4856720, 326.4540405, -531.9063721, 527.6434937
1: -229.4598083, 332.8937378, -240.6076508, 348.3619995, -577.8217773, 573.5013428
2: -233.1933289, 328.2497253, -244.4766083, 343.6737366, -576.8670044, 572.7263184
3: -280.7006531, 385.4885559, -294.2843933, 403.1745911, -683.8751831, 679.7729492
4: -255.1674500, 379.6665955, -267.0398254, 397.5583801, -652.7257080, 646.7064209

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2443541, upper bound: 495.2655846
time: 1.96 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2589608, upper bound: 495.2914396
time: 1.42 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -220.2614441, 334.0695190, -212.5932922, 323.4444580, -543.7058105, 546.6628418
1: -245.9608459, 356.2183838, -237.4012604, 344.7285461, -590.6893311, 593.6196289
2: -249.9733276, 351.4279785, -241.2593231, 339.8170166, -589.7903442, 592.6871338
3: -300.6258850, 412.4967651, -290.2413635, 399.4304810, -700.0563965, 702.7381592
4: -273.0295410, 406.6106567, -264.0952454, 392.8693237, -665.8988647, 670.7059326

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2877250, upper bound: 495.2858913
time: 1.84 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2884401, upper bound: 495.2866232
time: 1.27 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -220.2614441, 334.0695190, -205.4523468, 312.1578369, -532.4193115, 539.5218506
1: -245.9608459, 356.2183838, -229.4598083, 332.8937378, -578.8545532, 585.6782227
2: -249.9733276, 351.4279785, -233.1933289, 328.2497253, -578.2230225, 584.6212769
3: -300.6258850, 412.4967651, -280.7006531, 385.4885559, -686.1144409, 693.1973267
4: -273.0295410, 406.6106567, -255.1674500, 379.6665955, -652.6961670, 661.7780762

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2877250, upper bound: 495.2884203
time: 1.21 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2884401, upper bound: 495.2887365
time: 1.27 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -215.4856720, 326.4540405, -212.5932922, 323.4444580, -538.9301147, 539.0473022
1: -240.6076508, 348.3619995, -237.4012604, 344.7285461, -585.3361816, 585.7632446
2: -244.4766083, 343.6737366, -241.2593231, 339.8170166, -584.2936401, 584.9328613
3: -294.2843933, 403.1745911, -290.2413635, 399.4304810, -693.7148438, 693.4159546
4: -267.0398254, 397.5583801, -264.0952454, 392.8693237, -659.9091797, 661.6534424

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2772549, upper bound: 495.2230380
time: 1.33 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2884463, upper bound: 495.2564693
time: 1.24 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -215.4856720, 326.4540405, -205.4523468, 312.1578369, -527.6434937, 531.9063721
1: -240.6076508, 348.3619995, -229.4598083, 332.8937378, -573.5012817, 577.8217773
2: -244.4766083, 343.6737366, -233.1933289, 328.2497253, -572.7263184, 576.8670654
3: -294.2843933, 403.1745911, -280.7006531, 385.4885559, -679.7729492, 683.8751831
4: -267.0398254, 397.5583801, -255.1674500, 379.6665955, -646.7064209, 652.7256470

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2772549, upper bound: 495.2230380
time: 0.87 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2884463, upper bound: 495.2564693
time: 1.11 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -220.2614441, 334.0695190, -220.2614441, 334.0695190, -554.3309326, 554.3309326
1: -245.9608459, 356.2183838, -245.9608459, 356.2183838, -602.1791992, 602.1791992
2: -249.9733276, 351.4279785, -249.9733276, 351.4279785, -601.4012451, 601.4012451
3: -300.6258850, 412.4967651, -300.6258850, 412.4967651, -713.1226807, 713.1226807
4: -273.0295410, 406.6106567, -273.0295410, 406.6106567, -679.6401978, 679.6401978

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2556016, upper bound: 495.2858654
time: 1.06 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2564827, upper bound: 495.2866059
time: 1.07 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -220.2614441, 334.0695190, -215.4856720, 326.4540405, -546.7153931, 549.5551758
1: -245.9608459, 356.2183838, -240.6076508, 348.3619995, -594.3226929, 596.8260498
2: -249.9733276, 351.4279785, -244.4766083, 343.6737366, -593.6469727, 595.9045410
3: -300.6258850, 412.4967651, -294.2843933, 403.1745911, -703.8004761, 706.7811279
4: -273.0295410, 406.6106567, -267.0398254, 397.5583801, -670.5878906, 673.6505127

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2556016, upper bound: 495.2858715
time: 1.20 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2564827, upper bound: 495.2866167
time: 1.07 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -215.4856720, 326.4540405, -220.2614441, 334.0695190, -549.5551758, 546.7153931
1: -240.6076508, 348.3619995, -245.9608459, 356.2183838, -596.8259888, 594.3226929
2: -244.4766083, 343.6737366, -249.9733276, 351.4279785, -595.9046021, 593.6470337
3: -294.2843933, 403.1745911, -300.6258850, 412.4967651, -706.7811279, 703.8004761
4: -267.0398254, 397.5583801, -273.0295410, 406.6106567, -673.6505127, 670.5878906

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2443736, upper bound: 495.2230121
time: 1.01 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2565248, upper bound: 495.2564665
time: 1.10 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -215.4856720, 326.4540405, -215.4856720, 326.4540405, -541.9396973, 541.9396973
1: -240.6076508, 348.3619995, -240.6076508, 348.3619995, -588.9695435, 588.9696045
2: -244.4766083, 343.6737366, -244.4766083, 343.6737366, -588.1503296, 588.1503296
3: -294.2843933, 403.1745911, -294.2843933, 403.1745911, -697.4589844, 697.4589844
4: -267.0398254, 397.5583801, -267.0398254, 397.5583801, -664.5980835, 664.5980835

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2443736, upper bound: 495.2230121
time: 1.12 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2565248, upper bound: 495.2564665
time: 1.52 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 5.10 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.10
Output dim: 0, lower bound: -495.2862479, upper bound: 495.2858725
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.10
Output dim: 0, lower bound: -495.2884660, upper bound: 495.2884660
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.10
Output dim: 0, lower bound: -495.2862479, upper bound: 495.2884129
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.10
Output dim: 0, lower bound: -495.2884660, upper bound: 495.2909100
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.10
Output dim: 0, lower bound: -495.2838063, upper bound: 495.2656104
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.10
Output dim: 0, lower bound: -495.2909100, upper bound: 495.2914514
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.10
Output dim: 0, lower bound: -495.2838063, upper bound: 495.2656104
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.10
Output dim: 0, lower bound: -495.2909100, upper bound: 495.2940538
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.10
Output dim: 0, lower bound: -495.2511311, upper bound: 495.2858466
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.10
Output dim: 0, lower bound: -495.2564693, upper bound: 495.2884401
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.10
Output dim: 0, lower bound: -495.2511311, upper bound: 495.2858527
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.10
Output dim: 0, lower bound: -495.2564693, upper bound: 495.2884463
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.10
Output dim: 0, lower bound: -495.2443541, upper bound: 495.2655846
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.10
Output dim: 0, lower bound: -495.2589608, upper bound: 495.2914126
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.10
Output dim: 0, lower bound: -495.2443541, upper bound: 495.2655846
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.10
Output dim: 0, lower bound: -495.2589608, upper bound: 495.2914396
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.10
Output dim: 0, lower bound: -495.2877250, upper bound: 495.2858913
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.10
Output dim: 0, lower bound: -495.2884401, upper bound: 495.2866232
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.10
Output dim: 0, lower bound: -495.2877250, upper bound: 495.2884203
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.10
Output dim: 0, lower bound: -495.2884401, upper bound: 495.2887365
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.10
Output dim: 0, lower bound: -495.2772549, upper bound: 495.2230380
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.10
Output dim: 0, lower bound: -495.2884463, upper bound: 495.2564693
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.10
Output dim: 0, lower bound: -495.2772549, upper bound: 495.2230380
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.10
Output dim: 0, lower bound: -495.2884463, upper bound: 495.2564693
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.10
Output dim: 0, lower bound: -495.2556016, upper bound: 495.2858654
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.10
Output dim: 0, lower bound: -495.2564827, upper bound: 495.2866059
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.10
Output dim: 0, lower bound: -495.2556016, upper bound: 495.2858715
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.10
Output dim: 0, lower bound: -495.2564827, upper bound: 495.2866167
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.10
Output dim: 0, lower bound: -495.2443736, upper bound: 495.2230121
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.10
Output dim: 0, lower bound: -495.2565248, upper bound: 495.2564665
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.10
Output dim: 0, lower bound: -495.2443736, upper bound: 495.2230121
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.10
Output dim: 0, lower bound: -495.2565248, upper bound: 495.2564665

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -184.3121948, 280.2611084, -212.5932922, 323.4444580, -507.7566528, 492.8543091
1: -205.8135986, 298.8819885, -237.4012604, 344.7285461, -550.5421143, 536.2832031
2: -209.2110748, 294.6548767, -241.2593231, 339.8170166, -549.0280762, 535.9141235
3: -251.7965240, 346.4883118, -290.2413635, 399.4304810, -651.2269897, 636.7296753
4: -229.7188263, 340.5734558, -264.0952454, 392.8693237, -622.5881348, 604.6687012

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2838091, upper bound: 495.2838091
time: 1.63 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2838091, upper bound: 495.2858725
time: 1.41 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -208.4149323, 317.2074280, -212.5932922, 323.4444580, -531.8593750, 529.8007202
1: -232.7467041, 338.0377808, -237.4012604, 344.7285461, -577.4752197, 575.4390259
2: -236.5626373, 333.2463684, -241.2593231, 339.8170166, -576.3796387, 574.5055542
3: -284.5166016, 391.7001648, -290.2413635, 399.4304810, -683.9469604, 681.9415283
4: -259.0574036, 385.1869507, -264.0952454, 392.8693237, -651.9267578, 649.2821045

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2858725, upper bound: 495.2862479
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2858725, upper bound: 495.2884660
time: 1.47 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -184.3121948, 280.2611084, -205.4523468, 312.1578369, -496.4700317, 485.7134399
1: -205.8135986, 298.8819885, -229.4598083, 332.8937378, -538.7073364, 528.3417969
2: -209.2110748, 294.6548767, -233.1933289, 328.2497253, -537.4608154, 527.8482056
3: -251.7965240, 346.4883118, -280.7006531, 385.4885559, -637.2850952, 627.1889038
4: -229.7188263, 340.5734558, -255.1674500, 379.6665955, -609.3853760, 595.7409058

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2635471, upper bound: 495.2819262
time: 1.31 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2635471, upper bound: 495.2884129
time: 1.14 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -208.4149323, 317.2074280, -205.4523468, 312.1578369, -520.5727539, 522.6597900
1: -232.7467041, 338.0377808, -229.4598083, 332.8937378, -565.6404419, 567.4975586
2: -236.5626373, 333.2463684, -233.1933289, 328.2497253, -564.8123779, 566.4396973
3: -284.5166016, 391.7001648, -280.7006531, 385.4885559, -670.0050049, 672.4007568
4: -259.0574036, 385.1869507, -255.1674500, 379.6665955, -638.7239990, 640.3543091

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2656104, upper bound: 495.2838063
time: 1.07 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2656104, upper bound: 495.2909100
time: 1.63 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -179.2271881, 272.4698486, -212.5932922, 323.4444580, -502.6716309, 485.0630798
1: -200.1114044, 290.6095886, -237.4012604, 344.7285461, -544.8398438, 528.0108032
2: -203.4277344, 286.6937561, -241.2593231, 339.8170166, -543.2447510, 527.9528198
3: -244.9046173, 336.7877197, -290.2413635, 399.4304810, -644.3350830, 627.0290527
4: -223.3429413, 331.4250488, -264.0952454, 392.8693237, -616.2122803, 595.5202637

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2819262, upper bound: 495.2635471
time: 1.19 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2819262, upper bound: 495.2656104
time: 1.55 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -201.5160522, 306.2559204, -212.5932922, 323.4444580, -524.9604492, 518.8492432
1: -225.0784149, 326.5900269, -237.4012604, 344.7285461, -569.8069458, 563.9912720
2: -228.7740021, 322.0411072, -241.2593231, 339.8170166, -568.5910034, 563.3003540
3: -275.3319397, 378.2164001, -290.2413635, 399.4304810, -674.7622681, 668.4577637
4: -250.4766693, 372.4085083, -264.0952454, 392.8693237, -643.3460083, 636.5037231

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2884129, upper bound: 495.2886874
time: 1.72 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2884129, upper bound: 495.2914514
time: 1.88 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -179.2271881, 272.4698486, -205.4523468, 312.1578369, -491.3850098, 477.9221802
1: -200.1114044, 290.6095886, -229.4598083, 332.8937378, -533.0050659, 520.0693970
2: -203.4277344, 286.6937561, -233.1933289, 328.2497253, -531.6774902, 519.8870239
3: -244.9046173, 336.7877197, -280.7006531, 385.4885559, -630.3931885, 617.4882812
4: -223.3429413, 331.4250488, -255.1674500, 379.6665955, -603.0095215, 586.5925293

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2616917, upper bound: 495.2616917
time: 1.27 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2616917, upper bound: 495.2656104
time: 1.49 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -201.5160522, 306.2559204, -205.4523468, 312.1578369, -513.6738281, 511.7082520
1: -225.0784149, 326.5900269, -229.4598083, 332.8937378, -557.9721069, 556.0498047
2: -228.7740021, 322.0411072, -233.1933289, 328.2497253, -557.0237427, 555.2344360
3: -275.3319397, 378.2164001, -280.7006531, 385.4885559, -660.8203125, 658.9169312
4: -250.4766693, 372.4085083, -255.1674500, 379.6665955, -630.1432495, 627.5759277

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2678593, upper bound: 495.2851745
time: 2.47 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2678593, upper bound: 495.2940538
time: 1.83 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -184.3121948, 280.2611084, -220.2614441, 334.0695190, -518.3816528, 500.5225525
1: -205.8135986, 298.8819885, -245.9608459, 356.2183838, -562.0319824, 544.8427124
2: -209.2110748, 294.6548767, -249.9733276, 351.4279785, -560.6390381, 544.6281738
3: -251.7965240, 346.4883118, -300.6258850, 412.4967651, -664.2932739, 647.1141968
4: -229.7188263, 340.5734558, -273.0295410, 406.6106567, -636.3294678, 613.6030273

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2838279, upper bound: 495.2851315
time: 2.83 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2838279, upper bound: 495.2858466
time: 1.49 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -208.4149323, 317.2074280, -220.2614441, 334.0695190, -542.4844360, 537.4688110
1: -232.7467041, 338.0377808, -245.9608459, 356.2183838, -588.9650879, 583.9985962
2: -236.5626373, 333.2463684, -249.9733276, 351.4279785, -587.9906006, 583.2196655
3: -284.5166016, 391.7001648, -300.6258850, 412.4967651, -697.0132446, 692.3260498
4: -259.0574036, 385.1869507, -273.0295410, 406.6106567, -665.6680908, 658.2164917

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2858913, upper bound: 495.2877250
time: 1.21 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2858913, upper bound: 495.2884401
time: 1.64 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -184.3121948, 280.2611084, -215.4856720, 326.4540405, -510.7662354, 495.7467651
1: -205.8135986, 298.8819885, -240.6076508, 348.3619995, -554.1755371, 539.4895630
2: -209.2110748, 294.6548767, -244.4766083, 343.6737366, -552.8848267, 539.1314697
3: -251.7965240, 346.4883118, -294.2843933, 403.1745911, -654.9711304, 640.7727051
4: -229.7188263, 340.5734558, -267.0398254, 397.5583801, -627.2770996, 607.6132812

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2209747, upper bound: 495.2751339
time: 1.60 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2209747, upper bound: 495.2858527
time: 1.37 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -208.4149323, 317.2074280, -215.4856720, 326.4540405, -534.8689575, 532.6931152
1: -232.7467041, 338.0377808, -240.6076508, 348.3619995, -581.1086426, 578.6454468
2: -236.5626373, 333.2463684, -244.4766083, 343.6737366, -580.2363281, 577.7229614
3: -284.5166016, 391.7001648, -294.2843933, 403.1745911, -687.6911011, 685.9844360
4: -259.0574036, 385.1869507, -267.0398254, 397.5583801, -656.6157227, 652.2266846

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2230380, upper bound: 495.2772549
time: 1.79 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2230380, upper bound: 495.2884463
time: 1.72 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -179.2271881, 272.4698486, -220.2614441, 334.0695190, -513.2966309, 492.7312927
1: -200.1114044, 290.6095886, -245.9608459, 356.2183838, -556.3297119, 536.5703735
2: -203.4277344, 286.6937561, -249.9733276, 351.4279785, -554.8557129, 536.6669922
3: -244.9046173, 336.7877197, -300.6258850, 412.4967651, -657.4013672, 637.4135742
4: -223.3429413, 331.4250488, -273.0295410, 406.6106567, -629.9536133, 604.4545898

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2819280, upper bound: 495.2648694
time: 1.75 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2819280, upper bound: 495.2655846
time: 1.36 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -201.5160522, 306.2559204, -220.2614441, 334.0695190, -535.5855103, 526.5172729
1: -225.0784149, 326.5900269, -245.9608459, 356.2183838, -581.2968140, 572.5507812
2: -228.7740021, 322.0411072, -249.9733276, 351.4279785, -580.2019043, 572.0144043
3: -275.3319397, 378.2164001, -300.6258850, 412.4967651, -687.8285522, 678.8422852
4: -250.4766693, 372.4085083, -273.0295410, 406.6106567, -657.0873413, 645.4380493

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2884203, upper bound: 495.2906680
time: 1.36 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2884203, upper bound: 495.2914126
time: 1.12 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -179.2271881, 272.4698486, -215.4856720, 326.4540405, -505.6811829, 487.9555054
1: -200.1114044, 290.6095886, -240.6076508, 348.3619995, -548.4732056, 531.2171021
2: -203.4277344, 286.6937561, -244.4766083, 343.6737366, -547.1014404, 531.1702881
3: -244.9046173, 336.7877197, -294.2843933, 403.1745911, -648.0792236, 631.0721436
4: -223.3429413, 331.4250488, -267.0398254, 397.5583801, -620.9013062, 598.4648438

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2191193, upper bound: 495.2602352
time: 1.42 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2191193, upper bound: 495.2655846
time: 1.45 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -201.5160522, 306.2559204, -215.4856720, 326.4540405, -527.9700317, 521.7415771
1: -225.0784149, 326.5900269, -240.6076508, 348.3619995, -573.4403687, 567.1975708
2: -228.7740021, 322.0411072, -244.4766083, 343.6737366, -572.4476929, 566.5177002
3: -275.3319397, 378.2164001, -294.2843933, 403.1745911, -678.5064087, 672.5007935
4: -250.4766693, 372.4085083, -267.0398254, 397.5583801, -648.0349731, 639.4483032

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2257364, upper bound: 495.2798487
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2257364, upper bound: 495.2914396
time: 1.28 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -194.9475555, 296.0765686, -212.5932922, 323.4444580, -518.3920288, 508.6697693
1: -217.7083740, 315.8141785, -237.4012604, 344.7285461, -562.4368896, 553.2153931
2: -221.1859589, 311.5988464, -241.2593231, 339.8170166, -561.0029297, 552.8581543
3: -266.1375122, 365.8910522, -290.2413635, 399.4304810, -665.5679932, 656.1323853
4: -242.5342865, 360.2071228, -264.0952454, 392.8693237, -635.4036255, 624.3021851

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2851315, upper bound: 495.2838279
time: 1.17 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2851315, upper bound: 495.2858913
time: 1.25 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -215.8826752, 327.4780884, -212.5932922, 323.4444580, -539.3271484, 540.0714111
1: -241.0616608, 349.1610413, -237.4012604, 344.7285461, -585.7901611, 586.5623169
2: -245.0384369, 344.4467773, -241.2593231, 339.8170166, -584.8554688, 585.7060547
3: -294.6216736, 404.4009705, -290.2413635, 399.4304810, -694.0521240, 694.6423340
4: -267.7382812, 398.5161133, -264.0952454, 392.8693237, -660.6076050, 662.6111450

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2858466, upper bound: 495.2843602
time: 1.12 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2858466, upper bound: 495.2866232
time: 1.32 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -194.9475555, 296.0765686, -205.4523468, 312.1578369, -507.1053772, 501.5289001
1: -217.7083740, 315.8141785, -229.4598083, 332.8937378, -550.6021118, 545.2739868
2: -221.1859589, 311.5988464, -233.1933289, 328.2497253, -549.4356689, 544.7921753
3: -266.1375122, 365.8910522, -280.7006531, 385.4885559, -651.6260376, 646.5914917
4: -242.5342865, 360.2071228, -255.1674500, 379.6665955, -622.2008667, 615.3743896

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2648694, upper bound: 495.2819280
time: 1.61 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2648694, upper bound: 495.2884203
time: 1.07 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -215.8826752, 327.4780884, -205.4523468, 312.1578369, -528.0405273, 532.9304199
1: -241.0616608, 349.1610413, -229.4598083, 332.8937378, -573.9553833, 578.6208496
2: -245.0384369, 344.4467773, -233.1933289, 328.2497253, -573.2881470, 577.6401367
3: -294.6216736, 404.4009705, -280.7006531, 385.4885559, -680.1102295, 685.1014404
4: -267.7382812, 398.5161133, -255.1674500, 379.6665955, -647.4049072, 653.6833496

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2655846, upper bound: 495.2821089
time: 1.68 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2655846, upper bound: 495.2887365
time: 1.18 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -193.5258484, 294.0271912, -212.5932922, 323.4444580, -516.9703369, 506.6203308
1: -216.1171265, 313.6663818, -237.4012604, 344.7285461, -560.8457031, 551.0675659
2: -219.5807037, 309.4717712, -241.2593231, 339.8170166, -559.3976440, 550.7310181
3: -264.3647766, 363.1974487, -290.2413635, 399.4304810, -663.7952881, 653.4388428
4: -240.6031342, 357.6382141, -264.0952454, 392.8693237, -633.4724731, 621.7334595

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2751339, upper bound: 495.2209747
time: 1.25 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2751339, upper bound: 495.2230380
time: 1.18 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -209.5408325, 317.5305481, -212.5932922, 323.4444580, -532.9852905, 530.1237183
1: -233.9590302, 338.7438965, -237.4012604, 344.7285461, -578.6875610, 576.1451416
2: -237.7774963, 334.2637329, -241.2593231, 339.8170166, -577.5944824, 575.5228882
3: -286.1626892, 392.0972900, -290.2413635, 399.4304810, -685.5930786, 682.3386230
4: -259.8625488, 386.6659851, -264.0952454, 392.8693237, -652.7318726, 650.7611694

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2858527, upper bound: 495.2511311
time: 1.08 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2858527, upper bound: 495.2564693
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -193.5258484, 294.0271912, -205.4523468, 312.1578369, -505.6836853, 499.4794617
1: -216.1171265, 313.6663818, -229.4598083, 332.8937378, -549.0108643, 543.1262207
2: -219.5807037, 309.4717712, -233.1933289, 328.2497253, -547.8304443, 542.6651001
3: -264.3647766, 363.1974487, -280.7006531, 385.4885559, -649.8533325, 643.8980103
4: -240.6031342, 357.6382141, -255.1674500, 379.6665955, -620.2696533, 612.8056641

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2602352, upper bound: 495.2191193
time: 1.45 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2602352, upper bound: 495.2230380
time: 1.31 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -209.5408325, 317.5305481, -205.4523468, 312.1578369, -521.6986694, 522.9828491
1: -233.9590302, 338.7438965, -229.4598083, 332.8937378, -566.8527222, 568.2037354
2: -237.7774963, 334.2637329, -233.1933289, 328.2497253, -566.0272217, 567.4570312
3: -286.1626892, 392.0972900, -280.7006531, 385.4885559, -671.6511230, 672.7979126
4: -259.8625488, 386.6659851, -255.1674500, 379.6665955, -639.5291748, 641.8333740

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2655907, upper bound: 495.2443541
time: 1.30 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2655907, upper bound: 495.2564693
time: 1.37 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -194.9475555, 296.0765686, -220.2614441, 334.0695190, -529.0170898, 516.3380127
1: -217.7083740, 315.8141785, -245.9608459, 356.2183838, -573.9267578, 561.7749023
2: -221.1859589, 311.5988464, -249.9733276, 351.4279785, -572.6138916, 561.5721436
3: -266.1375122, 365.8910522, -300.6258850, 412.4967651, -678.6342773, 666.5169067
4: -242.5342865, 360.2071228, -273.0295410, 406.6106567, -649.1449585, 633.2366333

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2851502, upper bound: 495.2850962
time: 1.37 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2851502, upper bound: 495.2858654
time: 1.22 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -215.8826752, 327.4780884, -220.2614441, 334.0695190, -549.9522095, 547.7395020
1: -241.0616608, 349.1610413, -245.9608459, 356.2183838, -597.2800293, 595.1218872
2: -245.0384369, 344.4467773, -249.9733276, 351.4279785, -596.4663696, 594.4201050
3: -294.6216736, 404.4009705, -300.6258850, 412.4967651, -707.1184082, 705.0268555
4: -267.7382812, 398.5161133, -273.0295410, 406.6106567, -674.3489380, 671.5455933

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2858482, upper bound: 495.2853660
time: 1.17 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2858482, upper bound: 495.2866059
time: 1.27 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -194.9475555, 296.0765686, -215.4856720, 326.4540405, -521.4016113, 511.5622559
1: -217.7083740, 315.8141785, -240.6076508, 348.3619995, -566.0703735, 556.4217529
2: -221.1859589, 311.5988464, -244.4766083, 343.6737366, -564.8596802, 556.0754395
3: -266.1375122, 365.8910522, -294.2843933, 403.1745911, -669.3121338, 660.1753540
4: -242.5342865, 360.2071228, -267.0398254, 397.5583801, -640.0926514, 627.2467651

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2222970, upper bound: 495.2752094
time: 1.34 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2222970, upper bound: 495.2858715
time: 1.39 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -215.8826752, 327.4780884, -215.4856720, 326.4540405, -542.3367310, 542.9637451
1: -241.0616608, 349.1610413, -240.6076508, 348.3619995, -589.4235840, 589.7686768
2: -245.0384369, 344.4467773, -244.4766083, 343.6737366, -588.7121582, 588.9234009
3: -294.6216736, 404.4009705, -294.2843933, 403.1745911, -697.7962646, 698.6852417
4: -267.7382812, 398.5161133, -267.0398254, 397.5583801, -665.2965698, 665.5557861

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2230121, upper bound: 495.2757848
time: 1.18 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2230121, upper bound: 495.2866167
time: 1.55 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -193.5258484, 294.0271912, -220.2614441, 334.0695190, -527.5953369, 514.2886353
1: -216.1171265, 313.6663818, -245.9608459, 356.2183838, -572.3355103, 559.6270142
2: -219.5807037, 309.4717712, -249.9733276, 351.4279785, -571.0085449, 559.4450684
3: -264.3647766, 363.1974487, -300.6258850, 412.4967651, -676.8615723, 663.8233643
4: -240.6031342, 357.6382141, -273.0295410, 406.6106567, -647.2138062, 630.6677246

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2751339, upper bound: 495.2209747
time: 1.23 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2751339, upper bound: 495.2230121
time: 1.14 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -209.5408325, 317.5305481, -220.2614441, 334.0695190, -543.6103516, 537.7918091
1: -233.9590302, 338.7438965, -245.9608459, 356.2183838, -590.1774292, 584.7047119
2: -237.7774963, 334.2637329, -249.9733276, 351.4279785, -589.2054443, 584.2369995
3: -286.1626892, 392.0972900, -300.6258850, 412.4967651, -698.6593628, 692.7231445
4: -259.8625488, 386.6659851, -273.0295410, 406.6106567, -666.4732056, 659.6955566

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2858527, upper bound: 495.2511311
time: 1.28 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2858527, upper bound: 495.2564665
time: 1.62 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -193.5258484, 294.0271912, -215.4856720, 326.4540405, -519.9798584, 509.5128479
1: -216.1171265, 313.6663818, -240.6076508, 348.3619995, -564.4791260, 554.2738647
2: -219.5807037, 309.4717712, -244.4766083, 343.6737366, -563.2543335, 553.9483643
3: -264.3647766, 363.1974487, -294.2843933, 403.1745911, -667.5393677, 657.4818115
4: -240.6031342, 357.6382141, -267.0398254, 397.5583801, -638.1613770, 624.6780396

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -495.2176628, upper bound: 495.2176628
time: 1.77 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -495.2176628, upper bound: 495.2230121
time: 1.48 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -209.5408325, 317.5305481, -215.4856720, 326.4540405, -535.9948730, 533.0161133
1: -233.9590302, 338.7438965, -240.6076508, 348.3619995, -582.3209839, 579.3515625
2: -237.7774963, 334.2637329, -244.4766083, 343.6737366, -581.4512329, 578.7403564
3: -286.1626892, 392.0972900, -294.2843933, 403.1745911, -689.3372192, 686.3817139
4: -259.8625488, 386.6659851, -267.0398254, 397.5583801, -657.4208374, 653.7058105

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2230183, upper bound: 495.2442957
time: 1.31 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2230183, upper bound: 495.2564665
time: 1.18 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 5.09 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2838091, upper bound: 495.2838091
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2838091, upper bound: 495.2858725
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2858725, upper bound: 495.2862479
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2858725, upper bound: 495.2884660
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2635471, upper bound: 495.2819262
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2635471, upper bound: 495.2884129
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2656104, upper bound: 495.2838063
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2656104, upper bound: 495.2909100
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2819262, upper bound: 495.2635471
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2819262, upper bound: 495.2656104
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2884129, upper bound: 495.2886874
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2884129, upper bound: 495.2914514
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2616917, upper bound: 495.2616917
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2616917, upper bound: 495.2656104
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2678593, upper bound: 495.2851745
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2678593, upper bound: 495.2940538
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2838279, upper bound: 495.2851315
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2838279, upper bound: 495.2858466
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2858913, upper bound: 495.2877250
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2858913, upper bound: 495.2884401
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2209747, upper bound: 495.2751339
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2209747, upper bound: 495.2858527
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2230380, upper bound: 495.2772549
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2230380, upper bound: 495.2884463
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2819280, upper bound: 495.2648694
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2819280, upper bound: 495.2655846
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2884203, upper bound: 495.2906680
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2884203, upper bound: 495.2914126
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2191193, upper bound: 495.2602352
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2191193, upper bound: 495.2655846
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2257364, upper bound: 495.2798487
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2257364, upper bound: 495.2914396
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2851315, upper bound: 495.2838279
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2851315, upper bound: 495.2858913
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2858466, upper bound: 495.2843602
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2858466, upper bound: 495.2866232
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2648694, upper bound: 495.2819280
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2648694, upper bound: 495.2884203
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2655846, upper bound: 495.2821089
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2655846, upper bound: 495.2887365
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2751339, upper bound: 495.2209747
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2751339, upper bound: 495.2230380
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2858527, upper bound: 495.2511311
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2858527, upper bound: 495.2564693
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2602352, upper bound: 495.2191193
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2602352, upper bound: 495.2230380
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2655907, upper bound: 495.2443541
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2655907, upper bound: 495.2564693
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2851502, upper bound: 495.2850962
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2851502, upper bound: 495.2858654
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2858482, upper bound: 495.2853660
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2858482, upper bound: 495.2866059
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2222970, upper bound: 495.2752094
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2222970, upper bound: 495.2858715
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2230121, upper bound: 495.2757848
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2230121, upper bound: 495.2866167
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2751339, upper bound: 495.2209747
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2751339, upper bound: 495.2230121
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2858527, upper bound: 495.2511311
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2858527, upper bound: 495.2564665
IS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2176628, upper bound: 495.2176628
IS_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2176628, upper bound: 495.2230121
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2230183, upper bound: 495.2442957
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.09
Output dim: 0, lower bound: -495.2230183, upper bound: 495.2564665

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -184.3121948, 280.2611084, -184.3121948, 280.2611084, -464.5733032, 464.5733032
1: -205.8135986, 298.8819885, -205.8135986, 298.8819885, -504.6955566, 504.6955566
2: -209.2110748, 294.6548767, -209.2110748, 294.6548767, -503.8659668, 503.8659668
3: -251.7965240, 346.4883118, -251.7965240, 346.4883118, -598.2848511, 598.2848511
4: -229.7188263, 340.5734558, -229.7188263, 340.5734558, -570.2922974, 570.2922974

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2551940, upper bound: 495.2747650
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2792944, upper bound: 495.2792944
time: 1.46 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -184.3121948, 280.2611084, -208.4149323, 317.2074280, -501.5196228, 488.6760254
1: -205.8135986, 298.8819885, -232.7467041, 338.0377808, -543.8513794, 531.6286621
2: -209.2110748, 294.6548767, -236.5626373, 333.2463684, -542.4574585, 531.2175293
3: -251.7965240, 346.4883118, -284.5166016, 391.7001648, -643.4965820, 631.0048218
4: -229.7188263, 340.5734558, -259.0574036, 385.1869507, -614.9057007, 599.6308594

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2551940, upper bound: 495.2761624
time: 1.41 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2792944, upper bound: 495.2806792
time: 1.18 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -208.4149323, 317.2074280, -184.3121948, 280.2611084, -488.6760254, 501.5196228
1: -232.7467041, 338.0377808, -205.8135986, 298.8819885, -531.6286621, 543.8513794
2: -236.5626373, 333.2463684, -209.2110748, 294.6548767, -531.2175293, 542.4574585
3: -284.5166016, 391.7001648, -251.7965240, 346.4883118, -631.0048828, 643.4967041
4: -259.0574036, 385.1869507, -229.7188263, 340.5734558, -599.6308594, 614.9057007

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2742962, upper bound: 495.2813105
time: 1.70 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2806792, upper bound: 495.2814402
time: 1.13 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -208.4149323, 317.2074280, -208.4149323, 317.2074280, -525.6223755, 525.6223755
1: -232.7467041, 338.0377808, -232.7467041, 338.0377808, -570.7844849, 570.7844849
2: -236.5626373, 333.2463684, -236.5626373, 333.2463684, -569.8090210, 569.8090210
3: -284.5166016, 391.7001648, -284.5166016, 391.7001648, -676.2166748, 676.2166138
4: -259.0574036, 385.1869507, -259.0574036, 385.1869507, -644.2443237, 644.2443237

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2742962, upper bound: 495.2829746
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2806792, upper bound: 495.2828453
time: 1.29 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -184.3121948, 280.2611084, -179.2271881, 272.4698486, -456.7820435, 459.4882812
1: -205.8135986, 298.8819885, -200.1114044, 290.6095886, -496.4231567, 498.9933167
2: -209.2110748, 294.6548767, -203.4277344, 286.6937561, -495.9048157, 498.0826111
3: -251.7965240, 346.4883118, -244.9046173, 336.7877197, -588.5842285, 591.3929443
4: -229.7188263, 340.5734558, -223.3429413, 331.4250488, -561.1438599, 563.9163818

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2178232, upper bound: 495.2699321
time: 1.19 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2600658, upper bound: 495.2780262
time: 1.11 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -184.3121948, 280.2611084, -201.5160522, 306.2559204, -490.5681152, 481.7771606
1: -205.8135986, 298.8819885, -225.0784149, 326.5900269, -532.4036255, 523.9603882
2: -209.2110748, 294.6548767, -228.7740021, 322.0411072, -531.2521973, 523.4288330
3: -251.7965240, 346.4883118, -275.3319397, 378.2164001, -630.0129395, 621.8201294
4: -229.7188263, 340.5734558, -250.4766693, 372.4085083, -602.1272583, 591.0501099

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2178232, upper bound: 495.2792339
time: 0.99 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2600658, upper bound: 495.2839774
time: 1.18 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -208.4149323, 317.2074280, -179.2271881, 272.4698486, -480.8847656, 496.4346313
1: -232.7467041, 338.0377808, -200.1114044, 290.6095886, -523.3563232, 538.1491089
2: -236.5626373, 333.2463684, -203.4277344, 286.6937561, -523.2563477, 536.6740723
3: -284.5166016, 391.7001648, -244.9046173, 336.7877197, -621.3042603, 636.6047974
4: -259.0574036, 385.1869507, -223.3429413, 331.4250488, -590.4824219, 608.5299072

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2239381, upper bound: 495.2752815
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2614507, upper bound: 495.2798291
time: 1.11 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -208.4149323, 317.2074280, -201.5160522, 306.2559204, -514.6708374, 518.7233887
1: -232.7467041, 338.0377808, -225.0784149, 326.5900269, -559.3366699, 563.1162109
2: -236.5626373, 333.2463684, -228.7740021, 322.0411072, -558.6037598, 562.0203247
3: -284.5166016, 391.7001648, -275.3319397, 378.2164001, -662.7329102, 667.0319214
4: -259.0574036, 385.1869507, -250.4766693, 372.4085083, -631.4659424, 635.6636353

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2239381, upper bound: 495.2850851
time: 1.23 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2614507, upper bound: 495.2849660
time: 1.08 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -179.2271881, 272.4698486, -184.3121948, 280.2611084, -459.4882812, 456.7820435
1: -200.1114044, 290.6095886, -205.8135986, 298.8819885, -498.9933167, 496.4231567
2: -203.4277344, 286.6937561, -209.2110748, 294.6548767, -498.0826111, 495.9048157
3: -244.9046173, 336.7877197, -251.7965240, 346.4883118, -591.3929443, 588.5842285
4: -223.3429413, 331.4250488, -229.7188263, 340.5734558, -563.9163818, 561.1438599

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2645499, upper bound: 495.2474305
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2567370, upper bound: 495.2480160
time: 1.21 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -179.2271881, 272.4698486, -208.4149323, 317.2074280, -496.4346313, 480.8847656
1: -200.1114044, 290.6095886, -232.7467041, 338.0377808, -538.1491089, 523.3562622
2: -203.4277344, 286.6937561, -236.5626373, 333.2463684, -536.6740723, 523.2563477
3: -244.9046173, 336.7877197, -284.5166016, 391.7001648, -636.6047974, 621.3041992
4: -223.3429413, 331.4250488, -259.0574036, 385.1869507, -608.5299072, 590.4824219

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2645499, upper bound: 495.2474305
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2567370, upper bound: 495.2480160
time: 1.20 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -201.5160522, 306.2559204, -184.3121948, 280.2611084, -481.7771606, 490.5681152
1: -225.0784149, 326.5900269, -205.8135986, 298.8819885, -523.9603271, 532.4036255
2: -228.7740021, 322.0411072, -209.2110748, 294.6548767, -523.4288940, 531.2521973
3: -275.3319397, 378.2164001, -251.7965240, 346.4883118, -621.8201904, 630.0129395
4: -250.4766693, 372.4085083, -229.7188263, 340.5734558, -591.0501099, 602.1273193

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2840141, upper bound: 495.2843683
time: 1.78 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2839774, upper bound: 495.2843628
time: 1.21 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -201.5160522, 306.2559204, -208.4149323, 317.2074280, -518.7234497, 514.6708374
1: -225.0784149, 326.5900269, -232.7467041, 338.0377808, -563.1162109, 559.3367310
2: -228.7740021, 322.0411072, -236.5626373, 333.2463684, -562.0203247, 558.6037598
3: -275.3319397, 378.2164001, -284.5166016, 391.7001648, -667.0319824, 662.7329102
4: -250.4766693, 372.4085083, -259.0574036, 385.1869507, -635.6636353, 631.4659424

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2709689, upper bound: 495.2684949
time: 1.60 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2567370, upper bound: 495.2522982
time: 1.24 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -179.2271881, 272.4698486, -179.2271881, 272.4698486, -451.6970215, 451.6970215
1: -200.1114044, 290.6095886, -200.1114044, 290.6095886, -490.7209473, 490.7209473
2: -203.4277344, 286.6937561, -203.4277344, 286.6937561, -490.1214905, 490.1214905
3: -244.9046173, 336.7877197, -244.9046173, 336.7877197, -581.6923218, 581.6923218
4: -223.3429413, 331.4250488, -223.3429413, 331.4250488, -554.7680054, 554.7680054

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2599018, upper bound: 495.2474305
time: 1.43 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2520645, upper bound: 495.2480160
time: 1.37 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -179.2271881, 272.4698486, -201.5160522, 306.2559204, -485.4830933, 473.9859009
1: -200.1114044, 290.6095886, -225.0784149, 326.5900269, -526.7013550, 515.6879272
2: -203.4277344, 286.6937561, -228.7740021, 322.0411072, -525.4688721, 515.4677734
3: -244.9046173, 336.7877197, -275.3319397, 378.2164001, -623.1210327, 612.1195068
4: -223.3429413, 331.4250488, -250.4766693, 372.4085083, -595.7514648, 581.9017334

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2599018, upper bound: 495.2474305
time: 2.39 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2520645, upper bound: 495.2480160
time: 1.69 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -201.5160522, 306.2559204, -179.2271881, 272.4698486, -473.9859009, 485.4830933
1: -225.0784149, 326.5900269, -200.1114044, 290.6095886, -515.6879272, 526.7013550
2: -228.7740021, 322.0411072, -203.4277344, 286.6937561, -515.4677124, 525.4688721
3: -275.3319397, 378.2164001, -244.9046173, 336.7877197, -612.1195679, 623.1210327
4: -250.4766693, 372.4085083, -223.3429413, 331.4250488, -581.9017334, 595.7514648

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2255514, upper bound: 495.2770442
time: 1.47 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2643558, upper bound: 495.2812918
time: 1.55 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -201.5160522, 306.2559204, -201.5160522, 306.2559204, -507.7719727, 507.7719727
1: -225.0784149, 326.5900269, -225.0784149, 326.5900269, -551.6683960, 551.6683960
2: -228.7740021, 322.0411072, -228.7740021, 322.0411072, -550.8151245, 550.8151245
3: -275.3319397, 378.2164001, -275.3319397, 378.2164001, -653.5482178, 653.5482178
4: -250.4766693, 372.4085083, -250.4766693, 372.4085083, -622.8851929, 622.8851929

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2660462, upper bound: 495.2717094
time: 1.48 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2520645, upper bound: 495.2558182
time: 1.29 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -184.3121948, 280.2611084, -194.9475555, 296.0765686, -480.3887634, 475.2086487
1: -205.8135986, 298.8819885, -217.7083740, 315.8141785, -521.6278076, 516.5903320
2: -209.2110748, 294.6548767, -221.1859589, 311.5988464, -520.8099365, 515.8408203
3: -251.7965240, 346.4883118, -266.1375122, 365.8910522, -617.6875610, 612.6258545
4: -229.7188263, 340.5734558, -242.5342865, 360.2071228, -589.9257812, 583.1077271

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2551587, upper bound: 495.2755662
time: 1.22 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2792590, upper bound: 495.2800828
time: 1.24 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -184.3121948, 280.2611084, -215.8826752, 327.4780884, -511.7902527, 496.1437378
1: -205.8135986, 298.8819885, -241.0616608, 349.1610413, -554.9746094, 539.9436035
2: -209.2110748, 294.6548767, -245.0384369, 344.4467773, -553.6578369, 539.6932983
3: -251.7965240, 346.4883118, -294.6216736, 404.4009705, -656.1974487, 641.1099854
4: -229.7188263, 340.5734558, -267.7382812, 398.5161133, -628.2348022, 608.3117676

Time for backsubstitution: 2.44 seconds
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=571.2453002929688
rel_dist={0: [-495.2995013334902, 495.2995013334903]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2747516, upper bound: 495.2766075
time: 1.10 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2746553, upper bound: 495.2746553
time: 1.13 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.43 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.43
Output dim: 0, lower bound: -495.2747516, upper bound: 495.2766075
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.43
Output dim: 0, lower bound: -495.2746553, upper bound: 495.2746553

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -217.9161682, 331.5028992, -225.2498016, 341.9817505, -559.8979492, 556.7526855
1: -243.3218842, 353.3270569, -251.6062927, 364.5447388, -607.8665161, 604.9332275
2: -247.2664490, 348.2680054, -255.5475159, 359.3117371, -606.5781860, 603.8154907
3: -297.4977417, 409.3919373, -307.5393677, 422.3923340, -719.8900757, 716.9312744
4: -270.4958191, 402.7216187, -279.2279053, 415.6925354, -686.1882935, 681.9495239

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2746318, upper bound: 495.2746318
time: 1.71 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2746318, upper bound: 495.2746553
time: 1.30 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -225.6279755, 342.1831360, -224.3591309, 340.6464233, -566.2742310, 566.5422363
1: -251.9228973, 365.0004272, -250.6298828, 363.1114502, -615.0343628, 615.6303101
2: -256.0439148, 359.9935608, -254.5875092, 357.9270325, -613.9708862, 614.5809937
3: -307.8818359, 422.5994568, -306.3559265, 420.7380066, -728.6196899, 728.9553833
4: -279.6287537, 416.4534912, -278.1498718, 414.0597229, -693.6884766, 694.6032104

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2675813, upper bound: 495.2533982
time: 1.08 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2540846, upper bound: 495.2540846
time: 1.17 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.59 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.59
Output dim: 0, lower bound: -495.2746318, upper bound: 495.2746318
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.59
Output dim: 0, lower bound: -495.2746318, upper bound: 495.2746553
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.59
Output dim: 0, lower bound: -495.2675813, upper bound: 495.2533982
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.59
Output dim: 0, lower bound: -495.2540846, upper bound: 495.2540846

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -217.9161682, 331.5028992, -217.9161682, 331.5028992, -549.4190674, 549.4190674
1: -243.3218842, 353.3270569, -243.3218842, 353.3270569, -596.6488647, 596.6488037
2: -247.2664490, 348.2680054, -247.2664490, 348.2680054, -595.5344238, 595.5344238
3: -297.4977417, 409.3919373, -297.4977417, 409.3919373, -706.8896484, 706.8896484
4: -270.4958191, 402.7216187, -270.4958191, 402.7216187, -673.2174072, 673.2174072

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2514197, upper bound: 495.2685467
time: 1.25 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2546161, upper bound: 495.2713448
time: 1.57 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -217.9161682, 331.5028992, -225.6279755, 342.1831360, -560.0993042, 557.1308594
1: -243.3218842, 353.3270569, -251.9228973, 365.0004272, -608.3223267, 605.2498779
2: -247.2664490, 348.2680054, -256.0439148, 359.9935608, -607.2600098, 604.3118896
3: -297.4977417, 409.3919373, -307.8818359, 422.5994568, -720.0971680, 717.2738037
4: -270.4958191, 402.7216187, -279.6287537, 416.4534912, -686.9491577, 682.3503418

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2514197, upper bound: 495.2685467
time: 1.36 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2546161, upper bound: 495.2713448
time: 1.38 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -224.9297638, 341.1281433, -219.0165405, 332.5271301, -557.4569092, 560.1446533
1: -251.1466827, 363.8579102, -244.7026062, 354.4459534, -605.5925903, 608.5605469
2: -255.2541046, 358.8800354, -248.5536041, 349.4264221, -604.6803589, 607.4335938
3: -306.9366455, 421.2815552, -299.1197510, 410.7020874, -717.6386108, 720.4013062
4: -278.7688293, 415.1745605, -271.6973572, 404.1748047, -682.9436035, 686.8718872

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2498637, upper bound: 495.2226880
time: 1.31 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2675813, upper bound: 495.2533982
time: 1.46 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -223.0451508, 338.2022705, -211.8624268, 321.0991821, -544.1442871, 550.0646973
1: -249.0515442, 360.8396912, -236.6990662, 342.4568787, -591.5084229, 597.5387573
2: -253.1056976, 355.8741760, -240.4052734, 337.7487793, -590.8544312, 596.2794189
3: -304.3713074, 417.7056274, -289.6476135, 396.5525818, -700.9238892, 707.3532715
4: -276.4721069, 411.6592407, -262.6730042, 390.9145203, -667.3865967, 674.3322754

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2320465, upper bound: 495.2226945
time: 1.46 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2540846, upper bound: 495.2540846
time: 1.54 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 5.38 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.38
Output dim: 0, lower bound: -495.2514197, upper bound: 495.2685467
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.38
Output dim: 0, lower bound: -495.2546161, upper bound: 495.2713448
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.38
Output dim: 0, lower bound: -495.2514197, upper bound: 495.2685467
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.38
Output dim: 0, lower bound: -495.2546161, upper bound: 495.2713448
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.38
Output dim: 0, lower bound: -495.2498637, upper bound: 495.2226880
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.38
Output dim: 0, lower bound: -495.2675813, upper bound: 495.2533982
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.38
Output dim: 0, lower bound: -495.2320465, upper bound: 495.2226945
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.38
Output dim: 0, lower bound: -495.2540846, upper bound: 495.2540846

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -212.5932922, 323.4444580, -217.2519836, 330.4971924, -543.0904541, 540.6964111
1: -237.4012604, 344.7285461, -242.5831757, 352.2536316, -589.6549072, 587.3117065
2: -241.2593231, 339.8170166, -246.5167847, 347.2124023, -588.4716187, 586.3338013
3: -290.2413635, 399.4304810, -296.5927734, 408.1488647, -698.3902588, 696.0231934
4: -264.0952454, 392.8693237, -269.6970215, 401.4923401, -665.5875244, 662.5663452

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2590337, upper bound: 495.2676981
time: 1.50 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2773535, upper bound: 495.2771903
time: 1.63 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -205.4523468, 312.1578369, -215.2288208, 327.2562256, -532.7085571, 527.3865967
1: -229.4598083, 332.8937378, -240.3314209, 348.8588867, -578.3187256, 573.2250977
2: -233.1933289, 328.2497253, -244.2073669, 343.8748169, -577.0681152, 572.4570923
3: -280.7006531, 385.4885559, -293.8736877, 404.1477051, -684.8482056, 679.3622437
4: -255.1674500, 379.6665955, -267.1721497, 397.7210999, -652.8885498, 646.8386230

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2599626, upper bound: 495.2679489
time: 1.69 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2796064, upper bound: 495.2796064
time: 1.39 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -212.5932922, 323.4444580, -224.9297638, 341.1281433, -553.7214355, 548.3742065
1: -237.4012604, 344.7285461, -251.1466827, 363.8579102, -601.2591553, 595.8751831
2: -241.2593231, 339.8170166, -255.2541046, 358.8800354, -600.1392822, 595.0711060
3: -290.2413635, 399.4304810, -306.9366455, 421.2815552, -711.5228882, 706.3670654
4: -264.0952454, 392.8693237, -278.7688293, 415.1745605, -679.2696533, 671.6381836

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2227070, upper bound: 495.2505628
time: 1.52 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2514197, upper bound: 495.2685467
time: 1.52 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -205.4523468, 312.1578369, -223.0451508, 338.2022705, -543.6546021, 535.2030029
1: -229.4598083, 332.8937378, -249.0515442, 360.8396912, -590.2994995, 581.9452515
2: -233.1933289, 328.2497253, -253.1056976, 355.8741760, -589.0675049, 581.3554077
3: -280.7006531, 385.4885559, -304.3713074, 417.7056274, -698.4062500, 689.8598022
4: -255.1674500, 379.6665955, -276.4721069, 411.6592407, -666.8266602, 656.1386719

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2252154, upper bound: 495.2530415
time: 1.54 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2546161, upper bound: 495.2713448
time: 1.22 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -200.8029785, 305.1607056, -212.7076569, 323.0136414, -523.8166504, 517.8683472
1: -224.2043915, 325.5416870, -237.6668396, 344.2919617, -568.4963379, 563.2084961
2: -227.8195190, 321.1214905, -241.4614563, 339.4590454, -567.2785645, 562.5829468
3: -274.0705872, 377.0503845, -290.5170593, 399.0228271, -673.0933838, 667.5673828
4: -249.6923523, 371.1770935, -264.1734924, 392.4678040, -642.1601562, 635.3505859

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2498637, upper bound: 495.2226874
time: 1.54 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2498637, upper bound: 495.2226874
time: 1.41 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -220.7238312, 334.8034668, -217.3335876, 330.0128174, -550.7365723, 552.1370850
1: -246.4394989, 357.0434875, -242.8285522, 351.7498169, -598.1892700, 599.8719482
2: -250.5138702, 352.1828613, -246.6629181, 346.7774048, -597.2912598, 598.8457642
3: -301.1650391, 413.4631042, -296.8157654, 407.5916443, -708.7567139, 710.2787476
4: -273.6965637, 407.3980408, -269.6731873, 401.0730896, -674.7696533, 677.0711670

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2675813, upper bound: 495.2514197
time: 1.94 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2675813, upper bound: 495.2514197
time: 1.66 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -199.6360474, 303.4691162, -206.2750244, 312.6539307, -512.2899780, 509.7440796
1: -222.8961487, 323.7371216, -230.4708710, 333.4321899, -556.3283081, 554.2078857
2: -226.4866791, 319.3364258, -234.1135406, 328.8915100, -555.3781738, 553.4499512
3: -272.4848633, 374.8410645, -282.0052185, 386.1532288, -658.6380615, 656.8463135
4: -248.2336426, 369.0838623, -255.9910889, 380.5397339, -628.7733154, 625.0749512

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2320465, upper bound: 495.2226945
time: 1.87 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2320465, upper bound: 495.2226945
time: 1.37 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -218.8759613, 331.9312744, -209.9833221, 318.2900085, -537.1659546, 541.9146118
1: -244.3833313, 354.0837402, -234.6003113, 339.4472656, -583.8305664, 588.6840210
2: -248.4060974, 349.2362671, -238.2890625, 334.7951355, -583.2012329, 587.5253296
3: -298.6438599, 409.9485779, -287.0789795, 393.0743408, -691.7182007, 697.0275269
4: -271.4411621, 403.9472656, -260.4117126, 387.4597778, -658.9009399, 664.3590088

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2540846, upper bound: 495.2540846
time: 1.55 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2540846, upper bound: 495.2540846
time: 1.44 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 5.41 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.41
Output dim: 0, lower bound: -495.2590337, upper bound: 495.2676981
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.41
Output dim: 0, lower bound: -495.2773535, upper bound: 495.2771903
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.41
Output dim: 0, lower bound: -495.2599626, upper bound: 495.2679489
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.41
Output dim: 0, lower bound: -495.2796064, upper bound: 495.2796064
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.41
Output dim: 0, lower bound: -495.2227070, upper bound: 495.2505628
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.41
Output dim: 0, lower bound: -495.2514197, upper bound: 495.2685467
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.41
Output dim: 0, lower bound: -495.2252154, upper bound: 495.2530415
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.41
Output dim: 0, lower bound: -495.2546161, upper bound: 495.2713448
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.41
Output dim: 0, lower bound: -495.2498637, upper bound: 495.2226874
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.41
Output dim: 0, lower bound: -495.2498637, upper bound: 495.2226874
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.41
Output dim: 0, lower bound: -495.2675813, upper bound: 495.2514197
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.41
Output dim: 0, lower bound: -495.2675813, upper bound: 495.2514197
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.41
Output dim: 0, lower bound: -495.2320465, upper bound: 495.2226945
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.41
Output dim: 0, lower bound: -495.2320465, upper bound: 495.2226945
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.41
Output dim: 0, lower bound: -495.2540846, upper bound: 495.2540846
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.41
Output dim: 0, lower bound: -495.2540846, upper bound: 495.2540846

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -206.0225372, 313.6486816, -188.8715057, 287.1875000, -493.2099915, 502.5201721
1: -230.0718994, 334.2709656, -210.8758087, 306.2767029, -536.3485107, 545.1467896
2: -233.8955841, 329.5236511, -214.3768616, 301.9307251, -535.8262939, 543.9004517
3: -281.2122498, 387.4212341, -257.9938965, 354.9795837, -636.1918335, 645.4151611
4: -256.3013306, 380.7361755, -235.2425232, 349.0738525, -605.3751831, 615.9786987

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2588019, upper bound: 495.2671100
time: 1.17 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2588019, upper bound: 495.2676981
time: 1.11 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -211.0701904, 321.1720581, -213.2621155, 324.5459290, -535.6160889, 534.4342041
1: -235.7049561, 342.2923889, -238.1371765, 345.8713074, -581.5762939, 580.4295654
2: -239.5483704, 337.4201660, -242.0314789, 340.9427490, -580.4910889, 579.4515991
3: -288.1480713, 396.6198730, -291.1278381, 400.7708740, -688.9189453, 687.7476807
4: -262.2670288, 390.0578308, -264.8911133, 394.1586609, -656.4256592, 654.9489746

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2765507, upper bound: 495.2746523
time: 1.51 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2765507, upper bound: 495.2771903
time: 1.07 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -199.3240204, 302.9784546, -187.1094666, 284.4431152, -483.7671509, 490.0879211
1: -222.6311798, 323.0837097, -208.9102936, 303.3830261, -526.0142212, 531.9940186
2: -226.3198242, 318.6207275, -212.3732147, 299.1406250, -525.4603882, 530.9938965
3: -272.3736572, 374.1802063, -255.6082764, 351.5513611, -623.9250488, 629.7884521
4: -247.8601379, 368.3712158, -233.0646057, 345.8663635, -593.7265015, 601.4357910

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2580143, upper bound: 495.2580143
time: 1.63 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2580143, upper bound: 495.2679489
time: 1.66 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -203.7750092, 309.6410828, -211.3358917, 321.4390564, -525.2140503, 520.9769897
1: -227.5918427, 330.2014771, -235.9937286, 342.6221924, -570.2139282, 566.1950684
2: -231.3097229, 325.6036377, -239.8301544, 337.7475891, -569.0573120, 565.4336548
3: -278.4091187, 382.3798218, -288.5426331, 396.9332886, -675.3422852, 670.9224243
4: -253.1613922, 376.5696716, -262.4587402, 390.5560913, -643.7174683, 639.0284424

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2679489, upper bound: 495.2599626
time: 1.67 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2679489, upper bound: 495.2796064
time: 2.08 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -206.0225372, 313.6486816, -200.8029785, 305.1607056, -511.1831970, 514.4516602
1: -230.0718994, 334.2709656, -224.2043915, 325.5416870, -555.6133423, 558.4753418
2: -233.8955841, 329.5236511, -227.8195190, 321.1214905, -555.0170898, 557.3431396
3: -281.2122498, 387.4212341, -274.0705872, 377.0503845, -658.2626343, 661.4918213
4: -256.3013306, 380.7361755, -249.6923523, 371.1770935, -627.4783936, 630.4285278

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2207290, upper bound: 495.2493283
time: 1.42 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2207290, upper bound: 495.2505628
time: 1.08 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -211.0701904, 321.1720581, -220.7238312, 334.8034668, -545.8736572, 541.8958130
1: -235.7049561, 342.2923889, -246.4394989, 357.0434875, -592.7484131, 588.7318726
2: -239.5483704, 337.4201660, -250.5138702, 352.1828613, -591.7312012, 587.9340210
3: -288.1480713, 396.6198730, -301.1650391, 413.4631042, -701.6111450, 697.7849121
4: -262.2670288, 390.0578308, -273.6965637, 407.3980408, -669.6650391, 663.7543945

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2396720, upper bound: 495.2651668
time: 1.13 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2396720, upper bound: 495.2685467
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -199.3240204, 302.9784546, -199.6360474, 303.4691162, -502.7931519, 502.6145020
1: -222.6311798, 323.0837097, -222.8961487, 323.7371216, -546.3682861, 545.9798584
2: -226.3198242, 318.6207275, -226.4866791, 319.3364258, -545.6562500, 545.1074219
3: -272.3736572, 374.1802063, -272.4848633, 374.8410645, -647.2147217, 646.6649780
4: -247.8601379, 368.3712158, -248.2336426, 369.0838623, -616.9439697, 616.6048584

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2188632, upper bound: 495.2451002
time: 1.63 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2188632, upper bound: 495.2530415
time: 1.47 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -203.7750092, 309.6410828, -218.8759613, 331.9312744, -535.7062988, 528.5170288
1: -227.5918427, 330.2014771, -244.3833313, 354.0837402, -581.6753540, 574.5847168
2: -231.3097229, 325.6036377, -248.4060974, 349.2362671, -580.5460205, 574.0095215
3: -278.4091187, 382.3798218, -298.6438599, 409.9485779, -688.3576660, 681.0235596
4: -253.1613922, 376.5696716, -271.4411621, 403.9472656, -657.1086426, 648.0108643

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2319070, upper bound: 495.2493498
time: 1.80 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2319070, upper bound: 495.2713448
time: 1.12 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -200.8029785, 305.1607056, -206.0225372, 313.6486816, -514.4516602, 511.1832275
1: -224.2043915, 325.5416870, -230.0718994, 334.2709656, -558.4753418, 555.6134033
2: -227.8195190, 321.1214905, -233.8955841, 329.5236511, -557.3431396, 555.0170898
3: -274.0705872, 377.0503845, -281.2122498, 387.4212341, -661.4918213, 658.2626343
4: -249.6923523, 371.1770935, -256.3013306, 380.7361755, -630.4285278, 627.4783936

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2498637, upper bound: 495.2226874
time: 1.42 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2498637, upper bound: 495.2226874
time: 1.57 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -200.8029785, 305.1607056, -214.0988464, 324.7577209, -525.5606689, 519.2595215
1: -224.2043915, 325.5416870, -239.0764923, 346.2181091, -570.4224854, 564.6181030
2: -227.8195190, 321.1214905, -243.0222931, 341.5383301, -569.3578491, 564.1437378
3: -274.0705872, 377.0503845, -292.2184143, 401.0230713, -675.0936279, 669.2687988
4: -249.6923523, 371.1770935, -265.6088562, 395.1627808, -644.8551025, 636.7859497

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2498637, upper bound: 495.2226874
time: 1.66 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2498637, upper bound: 495.2226874
time: 1.36 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -220.7238312, 334.8034668, -211.0701904, 321.1720581, -541.8958130, 545.8736572
1: -246.4394989, 357.0434875, -235.7049561, 342.2923889, -588.7318726, 592.7484131
2: -250.5138702, 352.1828613, -239.5483704, 337.4201660, -587.9340210, 591.7312012
3: -301.1650391, 413.4631042, -288.1480713, 396.6198730, -697.7849121, 701.6112061
4: -273.6965637, 407.3980408, -262.2670288, 390.0578308, -663.7543945, 669.6650391

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2675813, upper bound: 495.2514197
time: 1.33 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2675813, upper bound: 495.2514197
time: 1.97 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -220.7238312, 334.8034668, -218.6318054, 331.6199951, -552.3436890, 553.4353027
1: -246.4394989, 357.0434875, -244.1370239, 353.5953674, -600.0348511, 601.1805420
2: -250.5138702, 352.1828613, -248.1386261, 348.8318176, -599.3457031, 600.3214722
3: -301.1650391, 413.4631042, -298.3876648, 409.4919434, -710.6569824, 711.8507080
4: -273.6965637, 407.3980408, -271.0665283, 403.5974426, -677.2940063, 678.4644775

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2675813, upper bound: 495.2514197
time: 1.74 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2675813, upper bound: 495.2514197
time: 1.62 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -199.6360474, 303.4691162, -199.3248138, 302.9795532, -502.6156006, 502.7939148
1: -222.8961487, 323.7371216, -222.6320190, 323.0848999, -545.9810791, 546.3690796
2: -226.4866791, 319.3364258, -226.3207092, 318.6218872, -545.1085815, 545.6570435
3: -272.4848633, 374.8410645, -272.3746948, 374.1816101, -646.6664429, 647.2156982
4: -248.2336426, 369.0838623, -247.8610687, 368.3725281, -616.6062012, 616.9448242

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2320465, upper bound: 495.2226945
time: 1.59 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2320465, upper bound: 495.2226874
time: 1.90 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -199.6360474, 303.4691162, -209.9028473, 317.8586426, -517.4946899, 513.3719482
1: -222.8961487, 323.7371216, -234.4018097, 339.0779114, -561.9740601, 558.1389160
2: -226.4866791, 319.3364258, -238.1844330, 334.5721130, -561.0587769, 557.5208740
3: -272.4848633, 374.8410645, -286.6624146, 392.5873413, -665.0722046, 661.5034790
4: -248.2336426, 369.0838623, -260.2488098, 387.1355286, -635.3690186, 629.3325806

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2320465, upper bound: 495.2226945
time: 1.32 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2320465, upper bound: 495.2226874
time: 1.51 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -218.8759613, 331.9312744, -203.7757874, 309.6422119, -528.5181885, 535.7070312
1: -244.3833313, 354.0837402, -227.5926971, 330.2026672, -574.5859375, 581.6763306
2: -248.4060974, 349.2362671, -231.3105927, 325.6048889, -574.0108032, 580.5467529
3: -298.6438599, 409.9485779, -278.4101257, 382.3812256, -681.0250854, 688.3587036
4: -271.4411621, 403.9472656, -253.1623230, 376.5711365, -648.0123291, 657.1095581

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2533982, upper bound: 495.2540846
time: 1.40 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2533982, upper bound: 495.2514197
time: 1.55 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -218.8759613, 331.9312744, -213.2514038, 323.1015625, -541.9774780, 545.1826782
1: -244.3833313, 354.0837402, -238.1113129, 344.7522583, -589.1356201, 592.1948853
2: -248.4060974, 349.2362671, -241.9601593, 340.1369934, -588.5430908, 591.1964111
3: -298.6438599, 409.9485779, -291.2317505, 399.0235901, -697.6674194, 701.1802979
4: -271.4411621, 403.9472656, -264.3543091, 393.4588013, -664.8999634, 668.3015747

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2533982, upper bound: 495.2540846
time: 1.47 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2533982, upper bound: 495.2514197
time: 1.68 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 5.62 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.62
Output dim: 0, lower bound: -495.2588019, upper bound: 495.2671100
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.62
Output dim: 0, lower bound: -495.2588019, upper bound: 495.2676981
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.62
Output dim: 0, lower bound: -495.2765507, upper bound: 495.2746523
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.62
Output dim: 0, lower bound: -495.2765507, upper bound: 495.2771903
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.62
Output dim: 0, lower bound: -495.2580143, upper bound: 495.2580143
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.62
Output dim: 0, lower bound: -495.2580143, upper bound: 495.2679489
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.62
Output dim: 0, lower bound: -495.2679489, upper bound: 495.2599626
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.62
Output dim: 0, lower bound: -495.2679489, upper bound: 495.2796064
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.62
Output dim: 0, lower bound: -495.2207290, upper bound: 495.2493283
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.62
Output dim: 0, lower bound: -495.2207290, upper bound: 495.2505628
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.62
Output dim: 0, lower bound: -495.2396720, upper bound: 495.2651668
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.62
Output dim: 0, lower bound: -495.2396720, upper bound: 495.2685467
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.62
Output dim: 0, lower bound: -495.2188632, upper bound: 495.2451002
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.62
Output dim: 0, lower bound: -495.2188632, upper bound: 495.2530415
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.62
Output dim: 0, lower bound: -495.2319070, upper bound: 495.2493498
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.62
Output dim: 0, lower bound: -495.2319070, upper bound: 495.2713448
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.62
Output dim: 0, lower bound: -495.2498637, upper bound: 495.2226874
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.62
Output dim: 0, lower bound: -495.2498637, upper bound: 495.2226874
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.62
Output dim: 0, lower bound: -495.2498637, upper bound: 495.2226874
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.62
Output dim: 0, lower bound: -495.2498637, upper bound: 495.2226874
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.62
Output dim: 0, lower bound: -495.2675813, upper bound: 495.2514197
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.62
Output dim: 0, lower bound: -495.2675813, upper bound: 495.2514197
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.62
Output dim: 0, lower bound: -495.2675813, upper bound: 495.2514197
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.62
Output dim: 0, lower bound: -495.2675813, upper bound: 495.2514197
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.62
Output dim: 0, lower bound: -495.2320465, upper bound: 495.2226945
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.62
Output dim: 0, lower bound: -495.2320465, upper bound: 495.2226874
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.62
Output dim: 0, lower bound: -495.2320465, upper bound: 495.2226945
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.62
Output dim: 0, lower bound: -495.2320465, upper bound: 495.2226874
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.62
Output dim: 0, lower bound: -495.2533982, upper bound: 495.2540846
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.62
Output dim: 0, lower bound: -495.2533982, upper bound: 495.2514197
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.62
Output dim: 0, lower bound: -495.2533982, upper bound: 495.2540846
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.62
Output dim: 0, lower bound: -495.2533982, upper bound: 495.2514197

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -184.3121948, 280.2611084, -188.8715057, 287.1875000, -471.4996643, 469.1325989
1: -205.8135986, 298.8819885, -210.8758087, 306.2767029, -512.0903320, 509.7577209
2: -209.2110748, 294.6548767, -214.3768616, 301.9307251, -511.1417847, 509.0317383
3: -251.7965240, 346.4883118, -257.9938965, 354.9795837, -606.7760620, 604.4821777
4: -229.7188263, 340.5734558, -235.2425232, 349.0738525, -578.7926636, 575.8159790

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2588019, upper bound: 495.2671100
time: 1.27 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2588019, upper bound: 495.2671100
time: 1.30 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -208.4149323, 317.2074280, -188.8715057, 287.1875000, -495.6024170, 506.0789185
1: -232.7467041, 338.0377808, -210.8758087, 306.2767029, -539.0234375, 548.9135742
2: -236.5626373, 333.2463684, -214.3768616, 301.9307251, -538.4933472, 547.6232300
3: -284.5166016, 391.7001648, -257.9938965, 354.9795837, -639.4959717, 649.6940918
4: -259.0574036, 385.1869507, -235.2425232, 349.0738525, -608.1312256, 620.4294434

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2588019, upper bound: 495.2676981
time: 1.79 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2588019, upper bound: 495.2676981
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -184.3121948, 280.2611084, -213.2621155, 324.5459290, -508.8581238, 493.5231934
1: -205.8135986, 298.8819885, -238.1371765, 345.8713074, -551.6849365, 537.0191040
2: -209.2110748, 294.6548767, -242.0314789, 340.9427490, -550.1538086, 536.6862793
3: -251.7965240, 346.4883118, -291.1278381, 400.7708740, -652.5673218, 637.6160889
4: -229.7188263, 340.5734558, -264.8911133, 394.1586609, -623.8775024, 605.4645996

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2588019, upper bound: 495.2746508
time: 2.16 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2588019, upper bound: 495.2746523
time: 1.49 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -208.4149323, 317.2074280, -213.2621155, 324.5459290, -532.9608765, 530.4695435
1: -232.7467041, 338.0377808, -238.1371765, 345.8713074, -578.6180420, 576.1749268
2: -236.5626373, 333.2463684, -242.0314789, 340.9427490, -577.5053711, 575.2777710
3: -284.5166016, 391.7001648, -291.1278381, 400.7708740, -685.2873535, 682.8278809
4: -259.0574036, 385.1869507, -264.8911133, 394.1586609, -653.2160645, 650.0780640

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2588019, upper bound: 495.2766622
time: 1.69 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2588019, upper bound: 495.2771903
time: 1.33 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -179.2271881, 272.4698486, -187.1094666, 284.4431152, -463.6702881, 459.5793152
1: -200.1114044, 290.6095886, -208.9102936, 303.3830261, -503.4943848, 499.5198975
2: -203.4277344, 286.6937561, -212.3732147, 299.1406250, -502.5683594, 499.0669250
3: -244.9046173, 336.7877197, -255.6082764, 351.5513611, -596.4559937, 592.3959961
4: -223.3429413, 331.4250488, -233.0646057, 345.8663635, -569.2092896, 564.4896240

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2580143, upper bound: 495.2580143
time: 1.51 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2580143, upper bound: 495.2580143
time: 1.67 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -201.5160522, 306.2559204, -187.1094666, 284.4431152, -485.9591675, 493.3653870
1: -225.0784149, 326.5900269, -208.9102936, 303.3830261, -528.4614258, 535.5003052
2: -228.7740021, 322.0411072, -212.3732147, 299.1406250, -527.9146118, 534.4142456
3: -275.3319397, 378.2164001, -255.6082764, 351.5513611, -626.8833008, 633.8246460
4: -250.4766693, 372.4085083, -233.0646057, 345.8663635, -596.3430176, 605.4731445

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2580143, upper bound: 495.2679489
time: 1.40 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2580143, upper bound: 495.2679489
time: 1.68 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -179.2271881, 272.4698486, -211.3358917, 321.4390564, -500.6662598, 483.8057251
1: -200.1114044, 290.6095886, -235.9937286, 342.6221924, -542.7334595, 526.6032715
2: -203.4277344, 286.6937561, -239.8301544, 337.7475891, -541.1752930, 526.5238647
3: -244.9046173, 336.7877197, -288.5426331, 396.9332886, -641.8378906, 625.3303223
4: -223.3429413, 331.4250488, -262.4587402, 390.5560913, -613.8990479, 593.8837891

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2580143, upper bound: 495.2590337
time: 1.45 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2580143, upper bound: 495.2590337
time: 1.40 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -201.5160522, 306.2559204, -211.3358917, 321.4390564, -522.9550781, 517.5917969
1: -225.0784149, 326.5900269, -235.9937286, 342.6221924, -567.7005615, 562.5836792
2: -228.7740021, 322.0411072, -239.8301544, 337.7475891, -566.5216064, 561.8712769
3: -275.3319397, 378.2164001, -288.5426331, 396.9332886, -672.2650146, 666.7589722
4: -250.4766693, 372.4085083, -262.4587402, 390.5560913, -641.0327759, 634.8672485

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2580143, upper bound: 495.2773534
time: 1.25 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2580143, upper bound: 495.2796064
time: 1.49 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -184.3121948, 280.2611084, -200.8029785, 305.1607056, -489.4729004, 481.0640869
1: -205.8135986, 298.8819885, -224.2043915, 325.5416870, -531.3552246, 523.0863647
2: -209.2110748, 294.6548767, -227.8195190, 321.1214905, -530.3325806, 522.4743652
3: -251.7965240, 346.4883118, -274.0705872, 377.0503845, -628.8469238, 620.5588989
4: -229.7188263, 340.5734558, -249.6923523, 371.1770935, -600.8959351, 590.2658081

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2207290, upper bound: 495.2493283
time: 1.77 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2207290, upper bound: 495.2493283
time: 1.31 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -208.4149323, 317.2074280, -200.8029785, 305.1607056, -513.5756226, 518.0103760
1: -232.7467041, 338.0377808, -224.2043915, 325.5416870, -558.2883301, 562.2421875
2: -236.5626373, 333.2463684, -227.8195190, 321.1214905, -557.6841431, 561.0659180
3: -284.5166016, 391.7001648, -274.0705872, 377.0503845, -661.5668335, 665.7706909
4: -259.0574036, 385.1869507, -249.6923523, 371.1770935, -630.2344971, 634.8792725

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2207290, upper bound: 495.2505628
time: 1.11 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2207290, upper bound: 495.2505628
time: 1.14 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -184.3121948, 280.2611084, -220.7238312, 334.8034668, -519.1156616, 500.9849243
1: -205.8135986, 298.8819885, -246.4394989, 357.0434875, -562.8569946, 545.3214111
2: -209.2110748, 294.6548767, -250.5138702, 352.1828613, -561.3939209, 545.1687622
3: -251.7965240, 346.4883118, -301.1650391, 413.4631042, -665.2596436, 647.6533203
4: -229.7188263, 340.5734558, -273.6965637, 407.3980408, -637.1168213, 614.2700195

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2207290, upper bound: 495.2651668
time: 1.14 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2207290, upper bound: 495.2651668
time: 1.97 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -208.4149323, 317.2074280, -220.7238312, 334.8034668, -543.2183838, 537.9312134
1: -232.7467041, 338.0377808, -246.4394989, 357.0434875, -589.7901611, 584.4772339
2: -236.5626373, 333.2463684, -250.5138702, 352.1828613, -588.7454834, 583.7602539
3: -284.5166016, 391.7001648, -301.1650391, 413.4631042, -697.9795532, 692.8651733
4: -259.0574036, 385.1869507, -273.6965637, 407.3980408, -666.4554443, 658.8834839

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2207290, upper bound: 495.2685467
time: 1.27 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2207290, upper bound: 495.2685467
time: 1.32 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -179.2271881, 272.4698486, -199.6360474, 303.4691162, -482.6962891, 472.1058960
1: -200.1114044, 290.6095886, -222.8961487, 323.7371216, -523.8484497, 513.5056763
2: -203.4277344, 286.6937561, -226.4866791, 319.3364258, -522.7641602, 513.1804199
3: -244.9046173, 336.7877197, -272.4848633, 374.8410645, -619.7456665, 609.2725830
4: -223.3429413, 331.4250488, -248.2336426, 369.0838623, -592.4268188, 579.6586914

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2188632, upper bound: 495.2451002
time: 1.31 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2188632, upper bound: 495.2451002
time: 1.69 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -201.5160522, 306.2559204, -199.6360474, 303.4691162, -504.9851685, 505.8919678
1: -225.0784149, 326.5900269, -222.8961487, 323.7371216, -548.8154907, 549.4862061
2: -228.7740021, 322.0411072, -226.4866791, 319.3364258, -548.1104126, 548.5277710
3: -275.3319397, 378.2164001, -272.4848633, 374.8410645, -650.1729736, 650.7012939
4: -250.4766693, 372.4085083, -248.2336426, 369.0838623, -619.5605469, 620.6420898

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2188632, upper bound: 495.2451002
time: 1.56 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2188632, upper bound: 495.2530415
time: 1.56 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -179.2271881, 272.4698486, -218.8759613, 331.9312744, -511.1584473, 491.3457642
1: -200.1114044, 290.6095886, -244.3833313, 354.0837402, -554.1949463, 534.9929199
2: -203.4277344, 286.6937561, -248.4060974, 349.2362671, -552.6640015, 535.0997925
3: -244.9046173, 336.7877197, -298.6438599, 409.9485779, -654.8532104, 635.4315186
4: -223.3429413, 331.4250488, -271.4411621, 403.9472656, -627.2902222, 602.8662109

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2188632, upper bound: 495.2493498
time: 1.55 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2188632, upper bound: 495.2493498
time: 1.79 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -201.5160522, 306.2559204, -218.8759613, 331.9312744, -533.4472656, 525.1318359
1: -225.0784149, 326.5900269, -244.3833313, 354.0837402, -579.1620483, 570.9733887
2: -228.7740021, 322.0411072, -248.4060974, 349.2362671, -578.0102539, 570.4472046
3: -275.3319397, 378.2164001, -298.6438599, 409.9485779, -685.2804565, 676.8602295
4: -250.4766693, 372.4085083, -271.4411621, 403.9472656, -654.4238892, 643.8496704

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2188632, upper bound: 495.2451002
time: 1.23 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2188632, upper bound: 495.2713448
time: 1.64 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -194.9475555, 296.0765686, -206.0225372, 313.6486816, -508.5962219, 502.0990295
1: -217.7083740, 315.8141785, -230.0718994, 334.2709656, -551.9793701, 545.8858643
2: -221.1859589, 311.5988464, -233.8955841, 329.5236511, -550.7094727, 545.4944458
3: -266.1375122, 365.8910522, -281.2122498, 387.4212341, -653.5587158, 647.1032715
4: -242.5342865, 360.2071228, -256.3013306, 380.7361755, -623.2704468, 616.5083618

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2493283, upper bound: 495.2207290
time: 1.20 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2493283, upper bound: 495.2227070
time: 1.33 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -193.4784851, 293.9510498, -206.0225372, 313.6486816, -507.1271667, 499.9735413
1: -216.0649872, 313.5864563, -230.0718994, 334.2709656, -550.3359375, 543.6583252
2: -219.5275574, 309.3925171, -233.8955841, 329.5236511, -549.0511475, 543.2880859
3: -264.3006287, 363.1055908, -281.2122498, 387.4212341, -651.7218628, 644.3178711
4: -240.5483551, 357.5468140, -256.3013306, 380.7361755, -621.2845459, 613.8480835

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2493283, upper bound: 495.2207290
time: 1.80 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2493283, upper bound: 495.2227070
time: 1.40 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -194.9475555, 296.0765686, -214.0988464, 324.7577209, -519.7052612, 510.1754150
1: -217.7083740, 315.8141785, -239.0764923, 346.2181091, -563.9265137, 554.8906250
2: -221.1859589, 311.5988464, -243.0222931, 341.5383301, -562.7241821, 554.6210938
3: -266.1375122, 365.8910522, -292.2184143, 401.0230713, -667.1605225, 658.1094971
4: -242.5342865, 360.2071228, -265.6088562, 395.1627808, -637.6970825, 625.8159180

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2493283, upper bound: 495.2207290
time: 1.61 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2493283, upper bound: 495.2226874
time: 1.52 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -193.4784851, 293.9510498, -214.0988464, 324.7577209, -518.2362061, 508.0498962
1: -216.0649872, 313.5864563, -239.0764923, 346.2181091, -562.2830811, 552.6629639
2: -219.5275574, 309.3925171, -243.0222931, 341.5383301, -561.0657349, 552.4147949
3: -264.3006287, 363.1055908, -292.2184143, 401.0230713, -665.3236084, 655.3239746
4: -240.5483551, 357.5468140, -265.6088562, 395.1627808, -635.7111206, 623.1556396

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2493283, upper bound: 495.2207290
time: 1.31 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2493283, upper bound: 495.2226874
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -215.8826752, 327.4780884, -211.0701904, 321.1720581, -537.0547485, 538.5482788
1: -241.0616608, 349.1610413, -235.7049561, 342.2923889, -583.3540039, 584.8659668
2: -245.0384369, 344.4467773, -239.5483704, 337.4201660, -582.4586182, 583.9951172
3: -294.6216736, 404.4009705, -288.1480713, 396.6198730, -691.2415771, 692.5490112
4: -267.7382812, 398.5161133, -262.2670288, 390.0578308, -657.7960815, 660.7830811

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2651668, upper bound: 495.2396720
time: 1.18 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2651668, upper bound: 495.2514197
time: 1.52 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -209.5408325, 317.5305481, -211.0701904, 321.1720581, -530.7128906, 528.6007080
1: -233.9590302, 338.7438965, -235.7049561, 342.2923889, -576.2514038, 574.4488525
2: -237.7774963, 334.2637329, -239.5483704, 337.4201660, -575.1976318, 573.8120728
3: -286.1626892, 392.0972900, -288.1480713, 396.6198730, -682.7825317, 680.2453613
4: -259.8625488, 386.6659851, -262.2670288, 390.0578308, -649.9204102, 648.9329834

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2651668, upper bound: 495.2396720
time: 1.34 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2651668, upper bound: 495.2514197
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -215.8826752, 327.4780884, -218.6318054, 331.6199951, -547.5026245, 546.1098633
1: -241.0616608, 349.1610413, -244.1370239, 353.5953674, -594.6569824, 593.2980957
2: -245.0384369, 344.4467773, -248.1386261, 348.8318176, -593.8702393, 592.5853882
3: -294.6216736, 404.4009705, -298.3876648, 409.4919434, -704.1136475, 702.7885132
4: -267.7382812, 398.5161133, -271.0665283, 403.5974426, -671.3356934, 669.5824585

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2651668, upper bound: 495.2396720
time: 1.26 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2651668, upper bound: 495.2514197
time: 1.16 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -209.5408325, 317.5305481, -218.6318054, 331.6199951, -541.1607666, 536.1622925
1: -233.9590302, 338.7438965, -244.1370239, 353.5953674, -587.5543823, 582.8809204
2: -237.7774963, 334.2637329, -248.1386261, 348.8318176, -586.6093140, 582.4022217
3: -286.1626892, 392.0972900, -298.3876648, 409.4919434, -695.6545410, 690.4849854
4: -259.8625488, 386.6659851, -271.0665283, 403.5974426, -663.4598999, 657.7324219

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2651668, upper bound: 495.2396720
time: 1.16 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2651668, upper bound: 495.2514197
time: 1.17 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -194.9475555, 296.0765686, -199.3248138, 302.9795532, -497.9271240, 495.4013367
1: -217.7083740, 315.8141785, -222.6320190, 323.0848999, -540.7932739, 538.4461670
2: -221.1859589, 311.5988464, -226.3207092, 318.6218872, -539.8078613, 537.9194946
3: -266.1375122, 365.8910522, -272.3746948, 374.1816101, -640.3190308, 638.2655029
4: -242.5342865, 360.2071228, -247.8610687, 368.3725281, -610.9067993, 608.0679932

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2451002, upper bound: 495.2188632
time: 1.10 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2451002, upper bound: 495.2252154
time: 1.30 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -193.4784851, 293.9510498, -199.3248138, 302.9795532, -496.4580383, 493.2758179
1: -216.0649872, 313.5864563, -222.6320190, 323.0848999, -539.1499023, 536.2185059
2: -219.5275574, 309.3925171, -226.3207092, 318.6218872, -538.1494141, 535.7132568
3: -264.3006287, 363.1055908, -272.3746948, 374.1816101, -638.4821167, 635.4801025
4: -240.5483551, 357.5468140, -247.8610687, 368.3725281, -608.9208984, 605.4077759

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2451002, upper bound: 495.2188632
time: 1.29 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2451002, upper bound: 495.2227070
time: 1.64 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -194.9475555, 296.0765686, -209.9028473, 317.8586426, -512.8061523, 505.9794312
1: -217.7083740, 315.8141785, -234.4018097, 339.0779114, -556.7862549, 550.2159424
2: -221.1859589, 311.5988464, -238.1844330, 334.5721130, -555.7580566, 549.7832642
3: -266.1375122, 365.8910522, -286.6624146, 392.5873413, -658.7248535, 652.5533447
4: -242.5342865, 360.2071228, -260.2488098, 387.1355286, -629.6697998, 620.4557495

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -495.2174075, upper bound: 495.2174075
time: 1.30 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -495.2174075, upper bound: 495.2226945
time: 1.21 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -193.4784851, 293.9510498, -209.9028473, 317.8586426, -511.3371277, 503.8538818
1: -216.0649872, 313.5864563, -234.4018097, 339.0779114, -555.1428833, 547.9882812
2: -219.5275574, 309.3925171, -238.1844330, 334.5721130, -554.0996704, 547.5769653
3: -264.3006287, 363.1055908, -286.6624146, 392.5873413, -656.8879395, 649.7679443
4: -240.5483551, 357.5468140, -260.2488098, 387.1355286, -627.6838379, 617.7955322

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -495.2174075, upper bound: 495.2174075
time: 1.85 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -495.2174075, upper bound: 495.2226874
time: 1.22 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -215.8826752, 327.4780884, -203.7757874, 309.6422119, -525.5249023, 531.2539062
1: -241.0616608, 349.1610413, -227.5926971, 330.2026672, -571.2642212, 576.7537231
2: -245.0384369, 344.4467773, -231.3105927, 325.6048889, -570.6431885, 575.7573242
3: -294.6216736, 404.4009705, -278.4101257, 382.3812256, -677.0029297, 682.8109741
4: -267.7382812, 398.5161133, -253.1623230, 376.5711365, -644.3093872, 651.6783447

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2493498, upper bound: 495.2319070
time: 1.33 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2493498, upper bound: 495.2546161
time: 1.69 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -209.5408325, 317.5305481, -203.7757874, 309.6422119, -519.1830444, 521.3062744
1: -233.9590302, 338.7438965, -227.5926971, 330.2026672, -564.1615601, 566.3365479
2: -237.7774963, 334.2637329, -231.3105927, 325.6048889, -563.3822632, 565.5741577
3: -286.1626892, 392.0972900, -278.4101257, 382.3812256, -668.5438843, 670.5074463
4: -259.8625488, 386.6659851, -253.1623230, 376.5711365, -636.4336548, 639.8283081

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2493498, upper bound: 495.2319070
time: 1.17 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2493498, upper bound: 495.2514197
time: 1.10 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -215.8826752, 327.4780884, -213.2514038, 323.1015625, -538.9841919, 540.7294922
1: -241.0616608, 349.1610413, -238.1113129, 344.7522583, -585.8139038, 587.2723389
2: -245.0384369, 344.4467773, -241.9601593, 340.1369934, -585.1754150, 586.4069214
3: -294.6216736, 404.4009705, -291.2317505, 399.0235901, -693.6452637, 695.6326904
4: -267.7382812, 398.5161133, -264.3543091, 393.4588013, -661.1970215, 662.8704224

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2226880, upper bound: 495.2319070
time: 1.51 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2226880, upper bound: 495.2540846
time: 1.18 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -209.5408325, 317.5305481, -213.2514038, 323.1015625, -532.6423950, 530.7818604
1: -233.9590302, 338.7438965, -238.1113129, 344.7522583, -578.7113037, 576.8551025
2: -237.7774963, 334.2637329, -241.9601593, 340.1369934, -577.9144897, 576.2238159
3: -286.1626892, 392.0972900, -291.2317505, 399.0235901, -685.1862183, 683.3290405
4: -259.8625488, 386.6659851, -264.3543091, 393.4588013, -653.3212891, 651.0202637

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2226880, upper bound: 495.2319070
time: 1.02 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2226880, upper bound: 495.2514197
time: 1.07 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.60 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2588019, upper bound: 495.2671100
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2588019, upper bound: 495.2671100
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2588019, upper bound: 495.2676981
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2588019, upper bound: 495.2676981
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2588019, upper bound: 495.2746508
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2588019, upper bound: 495.2746523
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2588019, upper bound: 495.2766622
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2588019, upper bound: 495.2771903
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2580143, upper bound: 495.2580143
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2580143, upper bound: 495.2580143
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2580143, upper bound: 495.2679489
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2580143, upper bound: 495.2679489
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2580143, upper bound: 495.2590337
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2580143, upper bound: 495.2590337
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2580143, upper bound: 495.2773534
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2580143, upper bound: 495.2796064
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2207290, upper bound: 495.2493283
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2207290, upper bound: 495.2493283
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2207290, upper bound: 495.2505628
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2207290, upper bound: 495.2505628
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2207290, upper bound: 495.2651668
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2207290, upper bound: 495.2651668
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2207290, upper bound: 495.2685467
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2207290, upper bound: 495.2685467
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2188632, upper bound: 495.2451002
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2188632, upper bound: 495.2451002
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2188632, upper bound: 495.2451002
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2188632, upper bound: 495.2530415
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2188632, upper bound: 495.2493498
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2188632, upper bound: 495.2493498
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2188632, upper bound: 495.2451002
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2188632, upper bound: 495.2713448
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2493283, upper bound: 495.2207290
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2493283, upper bound: 495.2227070
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2493283, upper bound: 495.2207290
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2493283, upper bound: 495.2227070
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2493283, upper bound: 495.2207290
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2493283, upper bound: 495.2226874
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2493283, upper bound: 495.2207290
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2493283, upper bound: 495.2226874
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2651668, upper bound: 495.2396720
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2651668, upper bound: 495.2514197
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2651668, upper bound: 495.2396720
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2651668, upper bound: 495.2514197
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2651668, upper bound: 495.2396720
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2651668, upper bound: 495.2514197
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2651668, upper bound: 495.2396720
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2651668, upper bound: 495.2514197
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2451002, upper bound: 495.2188632
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2451002, upper bound: 495.2252154
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2451002, upper bound: 495.2188632
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2451002, upper bound: 495.2227070
IS_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2174075, upper bound: 495.2174075
IS_A2_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2174075, upper bound: 495.2226945
IS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2174075, upper bound: 495.2174075
IS_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2174075, upper bound: 495.2226874
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2493498, upper bound: 495.2319070
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2493498, upper bound: 495.2546161
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2493498, upper bound: 495.2319070
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2493498, upper bound: 495.2514197
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2226880, upper bound: 495.2319070
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2226880, upper bound: 495.2540846
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2226880, upper bound: 495.2319070
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.60
Output dim: 0, lower bound: -495.2226880, upper bound: 495.2514197

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -184.3121948, 280.2611084, -184.3121948, 280.2611084, -464.5733032, 464.5733032
1: -205.8135986, 298.8819885, -205.8135986, 298.8819885, -504.6955566, 504.6955566
2: -209.2110748, 294.6548767, -209.2110748, 294.6548767, -503.8659668, 503.8659668
3: -251.7965240, 346.4883118, -251.7965240, 346.4883118, -598.2848511, 598.2848511
4: -229.7188263, 340.5734558, -229.7188263, 340.5734558, -570.2922974, 570.2922974

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.1669923, upper bound: 495.2311509
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2561037, upper bound: 495.2651660
time: 1.47 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -184.3121948, 280.2611084, -179.1979218, 272.4217529, -456.7338562, 459.4590149
1: -205.8135986, 298.8819885, -200.0785828, 290.5587463, -496.3723450, 498.9605408
2: -209.2110748, 294.6548767, -203.3939056, 286.6439819, -495.8550415, 498.0487671
3: -251.7965240, 346.4883118, -244.8622437, 336.7320557, -588.5285645, 591.3505859
4: -229.7188263, 340.5734558, -223.3089142, 331.3695374, -561.0881958, 563.8823853

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.1669923, upper bound: 495.2311509
time: 1.30 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2561037, upper bound: 495.2651660
time: 1.29 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -208.4149323, 317.2074280, -184.3121948, 280.2611084, -488.6760254, 501.5196228
1: -232.7467041, 338.0377808, -205.8135986, 298.8819885, -531.6286621, 543.8513794
2: -236.5626373, 333.2463684, -209.2110748, 294.6548767, -531.2175293, 542.4574585
3: -284.5166016, 391.7001648, -251.7965240, 346.4883118, -631.0048828, 643.4967041
4: -259.0574036, 385.1869507, -229.7188263, 340.5734558, -599.6308594, 614.9057007

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.1760892, upper bound: 495.2395493
time: 1.55 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2563376, upper bound: 495.2656184
time: 1.71 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -208.4149323, 317.2074280, -179.1979218, 272.4217529, -480.8366394, 496.4053345
1: -232.7467041, 338.0377808, -200.0785828, 290.5587463, -523.3054199, 538.1163330
2: -236.5626373, 333.2463684, -203.3939056, 286.6439819, -523.2066040, 536.6401978
3: -284.5166016, 391.7001648, -244.8622437, 336.7320557, -621.2485962, 636.5623779
4: -259.0574036, 385.1869507, -223.3089142, 331.3695374, -590.4267578, 608.4957886

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.1760892, upper bound: 495.2395493
time: 2.08 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2563376, upper bound: 495.2656184
time: 1.62 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -184.3121948, 280.2611084, -208.4149323, 317.2074280, -501.5196228, 488.6760254
1: -205.8135986, 298.8819885, -232.7467041, 338.0377808, -543.8513794, 531.6286621
2: -209.2110748, 294.6548767, -236.5626373, 333.2463684, -542.4574585, 531.2175293
3: -251.7965240, 346.4883118, -284.5166016, 391.7001648, -643.4965820, 631.0048218
4: -229.7188263, 340.5734558, -259.0574036, 385.1869507, -614.9057007, 599.6308594

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2556999, upper bound: 495.2657585
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2727888, upper bound: 495.2717422
time: 1.22 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -184.3121948, 280.2611084, -201.5160522, 306.2559204, -490.5681152, 481.7771606
1: -205.8135986, 298.8819885, -225.0784149, 326.5900269, -532.4036255, 523.9603882
2: -209.2110748, 294.6548767, -228.7740021, 322.0411072, -531.2521973, 523.4288330
3: -251.7965240, 346.4883118, -275.3319397, 378.2164001, -630.0129395, 621.8201294
4: -229.7188263, 340.5734558, -250.4766693, 372.4085083, -602.1272583, 591.0501099

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2556999, upper bound: 495.2657585
time: 1.72 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2727888, upper bound: 495.2719955
time: 1.34 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -208.4149323, 317.2074280, -208.4149323, 317.2074280, -525.6223755, 525.6223755
1: -232.7467041, 338.0377808, -232.7467041, 338.0377808, -570.7844849, 570.7844849
2: -236.5626373, 333.2463684, -236.5626373, 333.2463684, -569.8090210, 569.8090210
3: -284.5166016, 391.7001648, -284.5166016, 391.7001648, -676.2166748, 676.2166138
4: -259.0574036, 385.1869507, -259.0574036, 385.1869507, -644.2443237, 644.2443237

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2658673, upper bound: 495.2729717
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2678014, upper bound: 495.2737471
time: 1.46 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -208.4149323, 317.2074280, -201.5160522, 306.2559204, -514.6708374, 518.7233887
1: -232.7467041, 338.0377808, -225.0784149, 326.5900269, -559.3366699, 563.1162109
2: -236.5626373, 333.2463684, -228.7740021, 322.0411072, -558.6037598, 562.0203247
3: -284.5166016, 391.7001648, -275.3319397, 378.2164001, -662.7329102, 667.0319214
4: -259.0574036, 385.1869507, -250.4766693, 372.4085083, -631.4659424, 635.6636353

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2658673, upper bound: 495.2735835
time: 1.40 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2678014, upper bound: 495.2743744
time: 1.75 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -179.2271881, 272.4698486, -184.3121948, 280.2611084, -459.4882812, 456.7820435
1: -200.1114044, 290.6095886, -205.8135986, 298.8819885, -498.9933167, 496.4231567
2: -203.4277344, 286.6937561, -209.2110748, 294.6548767, -498.0826111, 495.9048157
3: -244.9046173, 336.7877197, -251.7965240, 346.4883118, -591.3929443, 588.5842285
4: -223.3429413, 331.4250488, -229.7188263, 340.5734558, -563.9163818, 561.1438599

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2561511, upper bound: 495.2504284
time: 1.32 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2517088, upper bound: 495.2514180
time: 1.38 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -179.2271881, 272.4698486, -179.1979218, 272.4217529, -451.6488953, 451.6677856
1: -200.1114044, 290.6095886, -200.0785828, 290.5587463, -490.6701355, 490.6881714
2: -203.4277344, 286.6937561, -203.3939056, 286.6439819, -490.0717163, 490.0876160
3: -244.9046173, 336.7877197, -244.8622437, 336.7320557, -581.6366577, 581.6499634
4: -223.3429413, 331.4250488, -223.3089142, 331.3695374, -554.7124023, 554.7339478

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2561511, upper bound: 495.2504284
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2517088, upper bound: 495.2514180
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -201.5160522, 306.2559204, -184.3121948, 280.2611084, -481.7771606, 490.5681152
1: -225.0784149, 326.5900269, -205.8135986, 298.8819885, -523.9603271, 532.4036255
2: -228.7740021, 322.0411072, -209.2110748, 294.6548767, -523.4288940, 531.2521973
3: -275.3319397, 378.2164001, -251.7965240, 346.4883118, -621.8201904, 630.0129395
4: -250.4766693, 372.4085083, -229.7188263, 340.5734558, -591.0501099, 602.1273193

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.1764512, upper bound: 495.2401851
time: 1.26 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2578468, upper bound: 495.2658017
time: 1.41 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -201.5160522, 306.2559204, -179.1979218, 272.4217529, -473.9377441, 485.4538269
1: -225.0784149, 326.5900269, -200.0785828, 290.5587463, -515.6370850, 526.6685791
2: -228.7740021, 322.0411072, -203.3939056, 286.6439819, -515.4179688, 525.4349365
3: -275.3319397, 378.2164001, -244.8622437, 336.7320557, -612.0639038, 623.0786133
4: -250.4766693, 372.4085083, -223.3089142, 331.3695374, -581.8460083, 595.7173462

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.1764512, upper bound: 495.2401851
time: 1.42 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2578468, upper bound: 495.2658017
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -179.2271881, 272.4698486, -208.4149323, 317.2074280, -496.4346313, 480.8847656
1: -200.1114044, 290.6095886, -232.7467041, 338.0377808, -538.1491089, 523.3562622
2: -203.4277344, 286.6937561, -236.5626373, 333.2463684, -536.6740723, 523.2563477
3: -244.9046173, 336.7877197, -284.5166016, 391.7001648, -636.6047974, 621.3041992
4: -223.3429413, 331.4250488, -259.0574036, 385.1869507, -608.5299072, 590.4824219

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2573437, upper bound: 495.2467961
time: 1.22 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2537731, upper bound: 495.2477279
time: 1.59 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -179.2271881, 272.4698486, -201.5160522, 306.2559204, -485.4830933, 473.9859009
1: -200.1114044, 290.6095886, -225.0784149, 326.5900269, -526.7013550, 515.6879272
2: -203.4277344, 286.6937561, -228.7740021, 322.0411072, -525.4688721, 515.4677734
3: -244.9046173, 336.7877197, -275.3319397, 378.2164001, -623.1210327, 612.1195068
4: -223.3429413, 331.4250488, -250.4766693, 372.4085083, -595.7514648, 581.9017334

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2573437, upper bound: 495.2467961
time: 1.39 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2537731, upper bound: 495.2477279
time: 1.93 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -201.5160522, 306.2559204, -208.4149323, 317.2074280, -518.7234497, 514.6708374
1: -225.0784149, 326.5900269, -232.7467041, 338.0377808, -563.1162109, 559.3367310
2: -228.7740021, 322.0411072, -236.5626373, 333.2463684, -562.0203247, 558.6037598
3: -275.3319397, 378.2164001, -284.5166016, 391.7001648, -667.0319824, 662.7329102
4: -250.4766693, 372.4085083, -259.0574036, 385.1869507, -635.6636353, 631.4659424

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2712095, upper bound: 495.2747922
time: 1.22 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2743487, upper bound: 495.2749557
time: 8.05 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 11.88 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 11.88
Output dim: 0, lower bound: -495.1669923, upper bound: 495.2311509
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 11.88
Output dim: 0, lower bound: -495.2561037, upper bound: 495.2651660
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 11.88
Output dim: 0, lower bound: -495.1669923, upper bound: 495.2311509
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 11.88
Output dim: 0, lower bound: -495.2561037, upper bound: 495.2651660
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 11.88
Output dim: 0, lower bound: -495.1760892, upper bound: 495.2395493
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 11.88
Output dim: 0, lower bound: -495.2563376, upper bound: 495.2656184
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 11.88
Output dim: 0, lower bound: -495.1760892, upper bound: 495.2395493
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 11.88
Output dim: 0, lower bound: -495.2563376, upper bound: 495.2656184
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 11.88
Output dim: 0, lower bound: -495.2556999, upper bound: 495.2657585
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 11.88
Output dim: 0, lower bound: -495.2727888, upper bound: 495.2717422
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 11.88
Output dim: 0, lower bound: -495.2556999, upper bound: 495.2657585
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 11.88
Output dim: 0, lower bound: -495.2727888, upper bound: 495.2719955
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 11.88
Output dim: 0, lower bound: -495.2658673, upper bound: 495.2729717
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 11.88
Output dim: 0, lower bound: -495.2678014, upper bound: 495.2737471
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 11.88
Output dim: 0, lower bound: -495.2658673, upper bound: 495.2735835
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 11.88
Output dim: 0, lower bound: -495.2678014, upper bound: 495.2743744
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 11.88
Output dim: 0, lower bound: -495.2561511, upper bound: 495.2504284
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 11.88
Output dim: 0, lower bound: -495.2517088, upper bound: 495.2514180
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 11.88
Output dim: 0, lower bound: -495.2561511, upper bound: 495.2504284
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 11.88
Output dim: 0, lower bound: -495.2517088, upper bound: 495.2514180
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 11.88
Output dim: 0, lower bound: -495.1764512, upper bound: 495.2401851
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 11.88
Output dim: 0, lower bound: -495.2578468, upper bound: 495.2658017
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 11.88
Output dim: 0, lower bound: -495.1764512, upper bound: 495.2401851
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 11.88
Output dim: 0, lower bound: -495.2578468, upper bound: 495.2658017
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 11.88
Output dim: 0, lower bound: -495.2573437, upper bound: 495.2467961
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 11.88
Output dim: 0, lower bound: -495.2537731, upper bound: 495.2477279
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 11.88
Output dim: 0, lower bound: -495.2573437, upper bound: 495.2467961
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 11.88
Output dim: 0, lower bound: -495.2537731, upper bound: 495.2477279
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 11.88
Output dim: 0, lower bound: -495.2712095, upper bound: 495.2747922
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 11.88
Output dim: 0, lower bound: -495.2743487, upper bound: 495.2749557
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 11.88
Output dim: 0, lower bound: -495.2580143, upper bound: 495.2796064
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 11.88
Output dim: 0, lower bound: -495.2207290, upper bound: 495.2493283
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 11.88
Output dim: 0, lower bound: -495.2207290, upper bound: 495.2493283
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 11.88
Output dim: 0, lower bound: -495.2207290, upper bound: 495.2505628
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 11.88
Output dim: 0, lower bound: -495.2207290, upper bound: 495.2505628
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 11.88
Output dim: 0, lower bound: -495.2207290, upper bound: 495.2651668
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 11.88
Output dim: 0, lower bound: -495.2207290, upper bound: 495.2651668
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 11.88
Output dim: 0, lower bound: -495.2207290, upper bound: 495.2685467
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 11.88
Output dim: 0, lower bound: -495.2207290, upper bound: 495.2685467
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 11.88
Output dim: 0, lower bound: -495.2188632, upper bound: 495.2451002
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 11.88
Output dim: 0, lower bound: -495.2188632, upper bound: 495.2451002
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 11.88
Output dim: 0, lower bound: -495.2188632, upper bound: 495.2451002
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 11.88
Output dim: 0, lower bound: -495.2188632, upper bound: 495.2530415
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 11.88
Output dim: 0, lower bound: -495.2188632, upper bound: 495.2493498
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 11.88
Output dim: 0, lower bound: -495.2188632, upper bound: 495.2493498
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 11.88
Output dim: 0, lower bound: -495.2188632, upper bound: 495.2451002
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 11.88
Output dim: 0, lower bound: -495.2188632, upper bound: 495.2713448
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 11.88
Output dim: 0, lower bound: -495.2493283, upper bound: 495.2207290
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 11.88
Output dim: 0, lower bound: -495.2493283, upper bound: 495.2227070
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 11.88
Output dim: 0, lower bound: -495.2493283, upper bound: 495.2207290
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 11.88
Output dim: 0, lower bound: -495.2493283, upper bound: 495.2227070
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 11.88
Output dim: 0, lower bound: -495.2493283, upper bound: 495.2207290
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 11.88
Output dim: 0, lower bound: -495.2493283, upper bound: 495.2226874
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 11.88
Output dim: 0, lower bound: -495.2493283, upper bound: 495.2207290
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 11.88
Output dim: 0, lower bound: -495.2493283, upper bound: 495.2226874
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 11.88
Output dim: 0, lower bound: -495.2651668, upper bound: 495.2396720
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 11.88
Output dim: 0, lower bound: -495.2651668, upper bound: 495.2514197
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 11.88
Output dim: 0, lower bound: -495.2651668, upper bound: 495.2396720
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 11.88
Output dim: 0, lower bound: -495.2651668, upper bound: 495.2514197
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 11.88
Output dim: 0, lower bound: -495.2651668, upper bound: 495.2396720
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 11.88
Output dim: 0, lower bound: -495.2651668, upper bound: 495.2514197
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 11.88
Output dim: 0, lower bound: -495.2651668, upper bound: 495.2396720
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 11.88
Output dim: 0, lower bound: -495.2651668, upper bound: 495.2514197
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 11.88
Output dim: 0, lower bound: -495.2451002, upper bound: 495.2188632
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 11.88
Output dim: 0, lower bound: -495.2451002, upper bound: 495.2252154
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 11.88
Output dim: 0, lower bound: -495.2451002, upper bound: 495.2188632
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 11.88
Output dim: 0, lower bound: -495.2451002, upper bound: 495.2227070
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 11.88
Output dim: 0, lower bound: -495.2493498, upper bound: 495.2319070
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 11.88
Output dim: 0, lower bound: -495.2493498, upper bound: 495.2546161
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 11.88
Output dim: 0, lower bound: -495.2493498, upper bound: 495.2319070
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 11.88
Output dim: 0, lower bound: -495.2493498, upper bound: 495.2514197
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 11.88
Output dim: 0, lower bound: -495.2226880, upper bound: 495.2319070
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 11.88
Output dim: 0, lower bound: -495.2226880, upper bound: 495.2540846
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 11.88
Output dim: 0, lower bound: -495.2226880, upper bound: 495.2319070
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 11.88
Output dim: 0, lower bound: -495.2226880, upper bound: 495.2514197
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=571.2453002929688
rel_dist={0: [-495.2824179308574, 495.28241793085726]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2611466, upper bound: 495.2618215
time: 1.56 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2610830, upper bound: 495.2610830
time: 2.02 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 3.78 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 3.78
Output dim: 0, lower bound: -495.2611466, upper bound: 495.2618215
IS_A2, status: Status.UNKNOWN, split count: 1, time: 3.78
Output dim: 0, lower bound: -495.2610830, upper bound: 495.2610830

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -217.9161682, 331.5028992, -222.3144073, 337.8009644, -555.7171631, 553.8173218
1: -243.3218842, 353.3270569, -248.2853241, 360.0689392, -603.3908081, 601.6123047
2: -247.2664490, 348.2680054, -252.2354584, 354.9055481, -602.1719971, 600.5034790
3: -297.4977417, 409.3919373, -303.4649963, 417.2050171, -714.7026367, 712.8569336
4: -270.4958191, 402.7216187, -275.7402649, 410.5083923, -681.0041504, 678.4619141

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2597932, upper bound: 495.2617992
time: 1.43 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2484994, upper bound: 495.2551422
time: 1.55 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -225.6279755, 342.1831360, -221.5415497, 336.5578918, -562.1858521, 563.7246704
1: -251.9228973, 365.0004272, -247.4665985, 358.7310791, -610.6539917, 612.4669800
2: -256.0439148, 359.9935608, -251.4512177, 353.6453552, -609.6892700, 611.4447632
3: -307.8818359, 422.5994568, -302.4840393, 415.6652222, -723.5470581, 725.0834961
4: -279.6287537, 416.4534912, -274.8080139, 409.0157471, -688.6444092, 691.2614746

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2530667, upper bound: 495.2481207
time: 1.51 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2483833, upper bound: 495.2483833
time: 1.52 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 5.55 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 5.55
Output dim: 0, lower bound: -495.2597932, upper bound: 495.2617992
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 5.55
Output dim: 0, lower bound: -495.2484994, upper bound: 495.2551422
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 5.55
Output dim: 0, lower bound: -495.2530667, upper bound: 495.2481207
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 5.55
Output dim: 0, lower bound: -495.2483833, upper bound: 495.2483833

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -215.4000244, 327.6949158, -217.0157166, 329.7641907, -545.1641846, 544.7105713
1: -240.5234833, 349.2640076, -242.4000092, 351.4938354, -592.0172729, 591.6639404
2: -244.4268646, 344.2722778, -246.2534790, 346.4790344, -590.9058838, 590.5257568
3: -294.0682373, 404.6852112, -296.2632141, 407.2700806, -701.3383179, 700.9484253
4: -267.4707947, 398.0655212, -269.3401489, 400.6954041, -668.1660156, 667.4055786

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2362467, upper bound: 495.2349294
time: 2.18 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2595523, upper bound: 495.2616653
time: 1.43 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -212.6777802, 323.2373047, -209.7316589, 318.2482300, -530.9260254, 532.9689941
1: -237.4913940, 344.6299438, -234.3103943, 339.4095459, -576.9009399, 578.9402466
2: -241.3022919, 339.7185364, -238.0228882, 334.6842957, -575.9865723, 577.7414551
3: -290.4311523, 399.1878052, -286.6876831, 393.0415344, -683.4726562, 685.8754883
4: -264.0188904, 392.9803162, -260.2204590, 387.2496338, -651.2685547, 653.2008057

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2240368, upper bound: 495.2330140
time: 1.49 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2484994, upper bound: 495.2551422
time: 1.46 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -223.0235901, 338.2458496, -216.2085419, 328.4313965, -551.4549561, 554.4544067
1: -249.0286407, 360.7371826, -241.5363159, 350.0577698, -599.0864258, 602.2734985
2: -253.0984802, 355.8386841, -245.4154358, 345.1382751, -598.2366943, 601.2541504
3: -304.3591003, 417.6847534, -295.2444153, 405.6235657, -709.9826660, 712.9291992
4: -276.4237976, 411.6815796, -268.3717957, 399.1223145, -675.5460815, 680.0533447

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2330654, upper bound: 495.2212443
time: 1.11 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2530667, upper bound: 495.2481207
time: 1.59 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -220.6257477, 334.5446777, -209.7057495, 317.9493408, -538.5750732, 544.2503662
1: -246.3595428, 356.9884338, -234.2742462, 339.0842285, -585.4437866, 591.2626953
2: -250.3514862, 352.0654602, -237.9857330, 334.4495544, -584.8009644, 590.0512085
3: -301.0806885, 413.1732483, -286.6326599, 392.6453247, -693.7257690, 699.8059082
4: -273.5205383, 407.2215271, -260.1173096, 387.0313721, -660.5518799, 667.3387451

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -495.2242807, upper bound: 495.2210111
time: 2.17 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2483833, upper bound: 495.2483833
time: 1.51 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 6.03 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.03
Output dim: 0, lower bound: -495.2362467, upper bound: 495.2349294
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.03
Output dim: 0, lower bound: -495.2595523, upper bound: 495.2616653
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.03
Output dim: 0, lower bound: -495.2240368, upper bound: 495.2330140
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.03
Output dim: 0, lower bound: -495.2484994, upper bound: 495.2551422
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.03
Output dim: 0, lower bound: -495.2330654, upper bound: 495.2212443
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.03
Output dim: 0, lower bound: -495.2530667, upper bound: 495.2481207
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 6.03
Output dim: 0, lower bound: -495.2242807, upper bound: 495.2210111
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.03
Output dim: 0, lower bound: -495.2483833, upper bound: 495.2483833

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -187.0099487, 284.3648987, -205.2319641, 312.0800476, -499.0898743, 489.5968628
1: -208.8090668, 303.2629395, -229.2779541, 332.6158447, -541.4249268, 532.5407715
2: -212.2682800, 298.9649048, -233.0342407, 327.9258118, -540.1940918, 531.9991455
3: -255.4627838, 351.5216980, -280.2116394, 385.6296692, -641.0924683, 631.7333374
4: -232.9798279, 345.6069946, -255.3168945, 378.8688660, -611.8486328, 600.9238281

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2362467, upper bound: 495.2349294
time: 1.61 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2362467, upper bound: 495.2349294
time: 2.14 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -211.3448639, 321.6441345, -214.1753845, 325.5221863, -536.8670654, 535.8194580
1: -236.0043030, 342.7737122, -239.2350616, 346.9443970, -582.9486694, 582.0087280
2: -239.8681030, 337.8988647, -243.0619202, 342.0075989, -581.8757324, 580.9608154
3: -288.5131226, 397.1838074, -292.3730469, 402.0214844, -690.5345459, 689.5568848
4: -262.5840759, 390.6114197, -265.9229736, 395.4579773, -658.0420532, 656.5343628

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2595523, upper bound: 495.2616653
time: 1.42 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2595523, upper bound: 495.2616653
time: 1.62 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -184.9821320, 281.2391357, -198.7275848, 301.6870728, -486.6691895, 479.9667358
1: -206.5414581, 299.9508057, -222.0198975, 321.7014771, -528.2429199, 521.9705811
2: -209.9575653, 295.8046875, -225.6495056, 317.3278503, -527.2852783, 521.4541626
3: -252.7239838, 347.5610046, -271.6861267, 372.6166077, -625.3405762, 619.2471313
4: -230.4448547, 342.0001831, -247.0692749, 366.9184875, -597.3633423, 589.0694580

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2240368, upper bound: 495.2330140
time: 1.13 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2240368, upper bound: 495.2330140
time: 1.74 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -208.9003906, 317.5848694, -206.7443390, 313.7634583, -522.6638184, 524.3291626
1: -233.2835236, 338.5751343, -230.9875336, 334.6129150, -567.8964233, 569.5626831
2: -237.0556030, 333.7640991, -234.6676331, 329.9702454, -567.0258179, 568.4317017
3: -285.2585449, 392.1876526, -282.6128235, 387.5036621, -672.7622070, 674.8004150
4: -259.4510803, 386.0161743, -256.6429749, 381.7417297, -641.1927490, 642.6591187

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2484994, upper bound: 495.2551422
time: 1.46 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2484994, upper bound: 495.2551422
time: 1.48 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -198.3792877, 301.3968811, -204.9758148, 311.4190369, -509.7983398, 506.3726501
1: -221.5172424, 321.5223999, -228.9655304, 331.8843994, -553.4016113, 550.4879150
2: -225.0731964, 317.1845093, -232.7613525, 327.3103638, -552.3835449, 549.9457397
3: -270.7886658, 372.4381104, -279.8590698, 384.8254700, -655.6141357, 652.2971191
4: -246.7339783, 366.6431274, -254.9244232, 378.1764221, -624.9104004, 621.5675049

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2330654, upper bound: 495.2212443
time: 1.61 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2330654, upper bound: 495.2212443
time: 1.17 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -218.7760010, 331.8544312, -213.0511932, 323.6977844, -542.4737549, 544.9056396
1: -244.2758026, 353.8512573, -238.0103607, 344.9815674, -589.2573853, 591.8616333
2: -248.3111725, 349.0697327, -241.8587799, 340.1518555, -588.4630127, 590.9285278
3: -298.5332336, 409.8158875, -290.9089966, 399.7660828, -698.2993164, 700.7248535
4: -271.2985840, 403.8258362, -264.5614014, 393.2837219, -664.5822754, 668.3872070

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2530667, upper bound: 495.2468101
time: 1.31 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2530667, upper bound: 495.2468101
time: 1.55 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -216.4837036, 328.3075867, -205.8044281, 312.1072388, -528.5907593, 534.1119995
1: -241.7239838, 350.2836914, -229.9235840, 332.8215332, -574.5454102, 580.2072754
2: -245.6847992, 345.4815063, -233.5901489, 328.3157959, -574.0006104, 579.0715332
3: -295.3939209, 405.4723511, -281.3085938, 385.4009705, -680.7948608, 686.7808838
4: -268.5242310, 399.5732422, -255.4134979, 379.8651733, -648.3894043, 654.9867554

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2483833, upper bound: 495.2483833
time: 1.57 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2483833, upper bound: 495.2483833
time: 1.38 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 5.36 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.36
Output dim: 0, lower bound: -495.2362467, upper bound: 495.2349294
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.36
Output dim: 0, lower bound: -495.2362467, upper bound: 495.2349294
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.36
Output dim: 0, lower bound: -495.2595523, upper bound: 495.2616653
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.36
Output dim: 0, lower bound: -495.2595523, upper bound: 495.2616653
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.36
Output dim: 0, lower bound: -495.2240368, upper bound: 495.2330140
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.36
Output dim: 0, lower bound: -495.2240368, upper bound: 495.2330140
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.36
Output dim: 0, lower bound: -495.2484994, upper bound: 495.2551422
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.36
Output dim: 0, lower bound: -495.2484994, upper bound: 495.2551422
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.36
Output dim: 0, lower bound: -495.2330654, upper bound: 495.2212443
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.36
Output dim: 0, lower bound: -495.2330654, upper bound: 495.2212443
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.36
Output dim: 0, lower bound: -495.2530667, upper bound: 495.2468101
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.36
Output dim: 0, lower bound: -495.2530667, upper bound: 495.2468101
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.36
Output dim: 0, lower bound: -495.2483833, upper bound: 495.2483833
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.36
Output dim: 0, lower bound: -495.2483833, upper bound: 495.2483833

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -187.0099487, 284.3648987, -200.6241760, 305.5495605, -492.5593872, 484.9890747
1: -208.8090668, 303.2629395, -224.0582123, 325.6266174, -534.4356689, 527.3211060
2: -212.2682800, 298.9649048, -227.8433228, 321.0184631, -533.2867432, 526.8082275
3: -255.4627838, 351.5216980, -273.8067932, 377.5590515, -633.0218506, 625.3284302
4: -232.9798279, 345.6069946, -249.8897247, 370.7220154, -603.7017212, 595.4966431

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2362467, upper bound: 495.2349294
time: 1.87 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2362467, upper bound: 495.2349294
time: 1.44 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -187.0099487, 284.3648987, -209.0874481, 317.2221375, -504.2319336, 493.4523315
1: -208.8090668, 303.2629395, -233.4739227, 338.1668091, -546.9758911, 536.7368164
2: -212.2682800, 298.9649048, -237.3757629, 333.5485840, -545.8168945, 536.3406982
3: -255.4627838, 351.5216980, -285.3761902, 391.8119202, -647.2747192, 636.8978882
4: -232.9798279, 345.6069946, -259.6111755, 385.8717346, -618.8515625, 605.2181396

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2362467, upper bound: 495.2349294
time: 2.04 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2362467, upper bound: 495.2349294
time: 1.67 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -211.3448639, 321.6441345, -209.9118958, 319.4434814, -530.7883301, 531.5559082
1: -236.0043030, 342.7737122, -234.4146271, 340.4376526, -576.4418945, 577.1882935
2: -239.8681030, 337.8988647, -238.2464905, 335.5987854, -575.4669189, 576.1453857
3: -288.5131226, 397.1838074, -286.5607605, 394.4780579, -682.9911499, 683.7445068
4: -262.5840759, 390.6114197, -260.8715515, 387.9282532, -650.5123291, 651.4829712

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2595523, upper bound: 495.2615523
time: 1.47 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2595523, upper bound: 495.2616653
time: 1.44 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -211.3448639, 321.6441345, -217.4052277, 329.7722473, -541.1171265, 539.0493164
1: -236.0043030, 342.7737122, -242.7656403, 351.6178894, -587.6221924, 585.5393677
2: -239.8681030, 337.8988647, -246.7559967, 346.8706360, -586.7387695, 584.6548462
3: -288.5131226, 397.1838074, -296.7024536, 407.2291260, -695.7422485, 693.8862305
4: -262.5840759, 390.6114197, -269.5888977, 401.3200073, -663.9040527, 660.2003174

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2595523, upper bound: 495.2615523
time: 1.52 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2595523, upper bound: 495.2616653
time: 1.54 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -184.9821320, 281.2391357, -194.2662811, 295.3895874, -480.3716736, 475.5054321
1: -206.5414581, 299.9508057, -216.9944458, 314.9669495, -521.5083618, 516.9451904
2: -209.9575653, 295.8046875, -220.6489716, 310.6629028, -520.6204224, 516.4536133
3: -252.7239838, 347.5610046, -265.4923401, 364.8368530, -617.5608521, 613.0533447
4: -230.4448547, 342.0001831, -241.8355408, 359.0351562, -589.4799805, 583.8356934

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2240368, upper bound: 495.2330140
time: 1.61 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2240368, upper bound: 495.2330140
time: 1.18 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -184.9821320, 281.2391357, -205.1715698, 310.5591125, -495.5411987, 486.4107056
1: -206.5414581, 299.9508057, -229.1400299, 331.2454529, -537.7869263, 529.0907593
2: -209.9575653, 295.8046875, -232.8510895, 326.8856812, -536.8432617, 528.6557617
3: -252.7239838, 347.5610046, -280.1968994, 383.6255188, -636.3494263, 627.7578735
4: -230.4448547, 342.0001831, -254.4911957, 378.2940674, -608.7388916, 596.4913940

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2240368, upper bound: 495.2330140
time: 1.33 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2240368, upper bound: 495.2330140
time: 1.54 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -208.9003906, 317.5848694, -202.6608429, 307.9693298, -516.8696899, 520.2455444
1: -233.2835236, 338.5751343, -226.3521118, 328.4177856, -561.7012939, 564.9272461
2: -237.0556030, 333.7640991, -230.0593262, 323.8460999, -560.9017334, 563.8233643
3: -285.2585449, 392.1876526, -276.8908386, 380.3218689, -665.5804443, 669.0784912
4: -259.4510803, 386.0161743, -251.8371582, 374.5178528, -633.9688110, 637.8533325

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2429454, upper bound: 495.2523544
time: 1.97 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2431245, upper bound: 495.2517639
time: 1.38 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -208.9003906, 317.5848694, -211.3913422, 320.3126831, -529.2130737, 528.9761963
1: -233.2835236, 338.5751343, -236.0324402, 341.7463684, -575.0299072, 574.6075439
2: -237.0556030, 333.7640991, -239.8650513, 337.1915894, -574.2471924, 573.6291504
3: -285.2585449, 392.1876526, -288.6858215, 395.5670471, -680.8255615, 680.8734741
4: -259.4510803, 386.0161743, -262.1174927, 390.0430908, -649.4941406, 648.1336060

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2429454, upper bound: 495.2523544
time: 1.89 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2431245, upper bound: 495.2517639
time: 1.64 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -198.3792877, 301.3968811, -200.5965271, 305.5084534, -503.8876953, 501.9933472
1: -221.5172424, 321.5223999, -224.0274048, 325.5827942, -547.1000366, 545.5498047
2: -225.0731964, 317.1845093, -227.8121033, 320.9750977, -546.0482788, 544.9965210
3: -270.7886658, 372.4381104, -273.7690430, 377.5088501, -648.2974854, 646.2070923
4: -246.7339783, 366.6431274, -249.8562622, 370.6703796, -617.4043579, 616.4993896

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2330654, upper bound: 495.2212443
time: 1.42 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2330654, upper bound: 495.2212443
time: 1.47 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -198.3792877, 301.3968811, -209.0874481, 317.2221375, -515.6013184, 510.4843140
1: -221.5172424, 321.5223999, -233.4739227, 338.1668091, -559.6840820, 554.9962769
2: -225.0731964, 317.1845093, -237.3757629, 333.5485840, -558.6217651, 554.5601807
3: -270.7886658, 372.4381104, -285.3761902, 391.8119202, -662.6005859, 657.8142700
4: -246.7339783, 366.6431274, -259.6111755, 385.8717346, -632.6057129, 626.2542725

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2330654, upper bound: 495.2212443
time: 1.30 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2330654, upper bound: 495.2212443
time: 1.17 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -218.7760010, 331.8544312, -209.9118958, 319.4434814, -538.2194824, 541.7661133
1: -244.2758026, 353.8512573, -234.4146271, 340.4376526, -584.7133789, 588.2658081
2: -248.3111725, 349.0697327, -238.2464905, 335.5987854, -583.9099731, 587.3162231
3: -298.5332336, 409.8158875, -286.5607605, 394.4780579, -693.0112915, 696.3764038
4: -271.2985840, 403.8258362, -260.8715515, 387.9282532, -659.2266846, 664.6973877

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2530667, upper bound: 495.2468101
time: 1.32 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2530667, upper bound: 495.2468101
time: 1.24 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -218.7760010, 331.8544312, -217.4052277, 329.7722473, -548.5482178, 549.2594604
1: -244.2758026, 353.8512573, -242.7656403, 351.6178894, -595.8936768, 596.6168823
2: -248.3111725, 349.0697327, -246.7559967, 346.8706360, -595.1818237, 595.8257446
3: -298.5332336, 409.8158875, -296.7024536, 407.2291260, -705.7623291, 706.5182495
4: -271.2985840, 403.8258362, -269.5888977, 401.3200073, -672.6183472, 673.4147339

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2530667, upper bound: 495.2468101
time: 1.44 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2530667, upper bound: 495.2468101
time: 1.69 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -216.4837036, 328.3075867, -202.6616211, 307.9705200, -524.4541016, 530.9691772
1: -241.7239838, 350.2836914, -226.3529816, 328.4190674, -570.1430054, 576.6366577
2: -245.6847992, 345.4815063, -230.0602264, 323.8472900, -569.5321045, 575.5417480
3: -295.3939209, 405.4723511, -276.8918762, 380.3233032, -675.7172241, 682.3642578
4: -268.5242310, 399.5732422, -251.8381042, 374.5191956, -643.0434570, 651.4113770

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2406128, upper bound: 495.2444009
time: 1.29 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2423252, upper bound: 495.2423252
time: 1.68 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -216.4837036, 328.3075867, -211.3942261, 320.3169556, -536.8005981, 539.7017822
1: -241.7239838, 350.2836914, -236.0356293, 341.7509155, -583.4748535, 586.3193359
2: -245.6847992, 345.4815063, -239.8683167, 337.1960754, -582.8807983, 585.3497925
3: -295.3939209, 405.4723511, -288.6897583, 395.5722656, -690.9661865, 694.1620483
4: -268.5242310, 399.5732422, -262.1208801, 390.0482483, -658.5725098, 661.6940918

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2406128, upper bound: 495.2444009
time: 1.31 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2423252, upper bound: 495.2423252
time: 1.62 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 5.41 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 0, lower bound: -495.2362467, upper bound: 495.2349294
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 0, lower bound: -495.2362467, upper bound: 495.2349294
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 0, lower bound: -495.2362467, upper bound: 495.2349294
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 0, lower bound: -495.2362467, upper bound: 495.2349294
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 0, lower bound: -495.2595523, upper bound: 495.2615523
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 0, lower bound: -495.2595523, upper bound: 495.2616653
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 0, lower bound: -495.2595523, upper bound: 495.2615523
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 0, lower bound: -495.2595523, upper bound: 495.2616653
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 0, lower bound: -495.2240368, upper bound: 495.2330140
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 0, lower bound: -495.2240368, upper bound: 495.2330140
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 0, lower bound: -495.2240368, upper bound: 495.2330140
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 0, lower bound: -495.2240368, upper bound: 495.2330140
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 0, lower bound: -495.2429454, upper bound: 495.2523544
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 0, lower bound: -495.2431245, upper bound: 495.2517639
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 0, lower bound: -495.2429454, upper bound: 495.2523544
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 0, lower bound: -495.2431245, upper bound: 495.2517639
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 0, lower bound: -495.2330654, upper bound: 495.2212443
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 0, lower bound: -495.2330654, upper bound: 495.2212443
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 0, lower bound: -495.2330654, upper bound: 495.2212443
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 0, lower bound: -495.2330654, upper bound: 495.2212443
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 0, lower bound: -495.2530667, upper bound: 495.2468101
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 0, lower bound: -495.2530667, upper bound: 495.2468101
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 0, lower bound: -495.2530667, upper bound: 495.2468101
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 0, lower bound: -495.2530667, upper bound: 495.2468101
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 0, lower bound: -495.2406128, upper bound: 495.2444009
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 0, lower bound: -495.2423252, upper bound: 495.2423252
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 0, lower bound: -495.2406128, upper bound: 495.2444009
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.41
Output dim: 0, lower bound: -495.2423252, upper bound: 495.2423252

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -184.3121948, 280.2611084, -200.6241760, 305.5495605, -489.8617554, 480.8852844
1: -205.8135986, 298.8819885, -224.0582123, 325.6266174, -531.4401855, 522.9401855
2: -209.2110748, 294.6548767, -227.8433228, 321.0184631, -530.2295532, 522.4981079
3: -251.7965240, 346.4883118, -273.8067932, 377.5590515, -629.3555908, 620.2951050
4: -229.7188263, 340.5734558, -249.8897247, 370.7220154, -600.4407349, 590.4631958

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2494817, upper bound: 495.2486410
time: 1.47 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2461071, upper bound: 495.2439295
time: 1.53 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -179.1979218, 272.4217529, -200.6241760, 305.5495605, -484.7474365, 473.0458374
1: -200.0785828, 290.5587463, -224.0582123, 325.6266174, -525.7052002, 514.6169434
2: -203.3939056, 286.6439819, -227.8433228, 321.0184631, -524.4122925, 514.4871216
3: -244.8622437, 336.7320557, -273.8067932, 377.5590515, -622.4212646, 610.5388184
4: -223.3089142, 331.3695374, -249.8897247, 370.7220154, -594.0307617, 581.2590942

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2494817, upper bound: 495.2486410
time: 1.38 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2461071, upper bound: 495.2439295
time: 1.64 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -184.3121948, 280.2611084, -209.0874481, 317.2221375, -501.5342712, 489.3485413
1: -205.8135986, 298.8819885, -233.4739227, 338.1668091, -543.9804077, 532.3557739
2: -209.2110748, 294.6548767, -237.3757629, 333.5485840, -542.7596436, 532.0305176
3: -251.7965240, 346.4883118, -285.3761902, 391.8119202, -643.6084595, 631.8645020
4: -229.7188263, 340.5734558, -259.6111755, 385.8717346, -615.5905762, 600.1846313

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2362467, upper bound: 495.2349294
time: 1.48 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2362467, upper bound: 495.2349294
time: 1.79 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -179.1979218, 272.4217529, -209.0874481, 317.2221375, -496.4199829, 481.5091248
1: -200.0785828, 290.5587463, -233.4739227, 338.1668091, -538.2453613, 524.0325928
2: -203.3939056, 286.6439819, -237.3757629, 333.5485840, -536.9424438, 524.0196533
3: -244.8622437, 336.7320557, -285.3761902, 391.8119202, -636.6741943, 622.1082764
4: -223.3089142, 331.3695374, -259.6111755, 385.8717346, -609.1806641, 590.9805908

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2362467, upper bound: 495.2349294
time: 1.31 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2362467, upper bound: 495.2349294
time: 1.25 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -208.4149323, 317.2074280, -209.9118958, 319.4434814, -527.8583984, 527.1191406
1: -232.7467041, 338.0377808, -234.4146271, 340.4376526, -573.1842651, 572.4523926
2: -236.5626373, 333.2463684, -238.2464905, 335.5987854, -572.1614380, 571.4928589
3: -284.5166016, 391.7001648, -286.5607605, 394.4780579, -678.9945068, 678.2608643
4: -259.0574036, 385.1869507, -260.8715515, 387.9282532, -646.9855957, 646.0584717

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2583622, upper bound: 495.2556043
time: 1.44 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2605428, upper bound: 495.2605428
time: 1.51 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -201.5160522, 306.2559204, -209.9118958, 319.4434814, -520.9594727, 516.1677246
1: -225.0784149, 326.5900269, -234.4146271, 340.4376526, -565.5159912, 561.0045776
2: -228.7740021, 322.0411072, -238.2464905, 335.5987854, -564.3728027, 560.2875977
3: -275.3319397, 378.2164001, -286.5607605, 394.4780579, -669.8098145, 664.7770996
4: -250.4766693, 372.4085083, -260.8715515, 387.9282532, -638.4049072, 633.2800293

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2583622, upper bound: 495.2556043
time: 1.41 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2605428, upper bound: 495.2609945
time: 1.14 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -208.4149323, 317.2074280, -217.4052277, 329.7722473, -538.1871948, 534.6125488
1: -232.7467041, 338.0377808, -242.7656403, 351.6178894, -584.3646240, 580.8034058
2: -236.5626373, 333.2463684, -246.7559967, 346.8706360, -583.4332886, 580.0023804
3: -284.5166016, 391.7001648, -296.7024536, 407.2291260, -691.7457275, 688.4025879
4: -259.0574036, 385.1869507, -269.5888977, 401.3200073, -660.3772583, 654.7758179

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2544628, upper bound: 495.2535120
time: 1.47 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2573253, upper bound: 495.2593665
time: 1.37 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -201.5160522, 306.2559204, -217.4052277, 329.7722473, -531.2882690, 523.6610107
1: -225.0784149, 326.5900269, -242.7656403, 351.6178894, -576.6962891, 569.3556519
2: -228.7740021, 322.0411072, -246.7559967, 346.8706360, -575.6446533, 568.7971191
3: -275.3319397, 378.2164001, -296.7024536, 407.2291260, -682.5610352, 674.9188232
4: -250.4766693, 372.4085083, -269.5888977, 401.3200073, -651.7965088, 641.9974365

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2544628, upper bound: 495.2537136
time: 1.38 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2573253, upper bound: 495.2598026
time: 1.57 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -184.3121948, 280.2611084, -194.2662811, 295.3895874, -479.7017517, 474.5274048
1: -205.8135986, 298.8819885, -216.9944458, 314.9669495, -520.7805176, 515.8764648
2: -209.2110748, 294.6548767, -220.6489716, 310.6629028, -519.8739624, 515.3037720
3: -251.7965240, 346.4883118, -265.4923401, 364.8368530, -616.6333618, 611.9806519
4: -229.7188263, 340.5734558, -241.8355408, 359.0351562, -588.7539062, 582.4089966

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2494301, upper bound: 495.2487624
time: 1.76 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2461035, upper bound: 495.2460788
time: 1.74 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -179.1979218, 272.4217529, -194.2662811, 295.3895874, -474.5874634, 466.6880188
1: -200.0785828, 290.5587463, -216.9944458, 314.9669495, -515.0455322, 507.5531921
2: -203.3939056, 286.6439819, -220.6489716, 310.6629028, -514.0567017, 507.2929382
3: -244.8622437, 336.7320557, -265.4923401, 364.8368530, -609.6990967, 602.2243652
4: -223.3089142, 331.3695374, -241.8355408, 359.0351562, -582.3439331, 573.2049561

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2494301, upper bound: 495.2486410
time: 1.43 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2461035, upper bound: 495.2439295
time: 1.41 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -184.3121948, 280.2611084, -205.1715698, 310.5591125, -494.8712769, 485.4326782
1: -205.8135986, 298.8819885, -229.1400299, 331.2454529, -537.0590820, 528.0220337
2: -209.2110748, 294.6548767, -232.8510895, 326.8856812, -536.0967407, 527.5059204
3: -251.7965240, 346.4883118, -280.1968994, 383.6255188, -635.4219971, 626.6851807
4: -229.7188263, 340.5734558, -254.4911957, 378.2940674, -608.0128784, 595.0646362

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2183387, upper bound: 495.2304941
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2183387, upper bound: 495.2330140
time: 1.22 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -179.1979218, 272.4217529, -205.1715698, 310.5591125, -489.7569580, 477.5932922
1: -200.0785828, 290.5587463, -229.1400299, 331.2454529, -531.3240356, 519.6987305
2: -203.3939056, 286.6439819, -232.8510895, 326.8856812, -530.2794800, 519.4949341
3: -244.8622437, 336.7320557, -280.1968994, 383.6255188, -628.4876709, 616.9289551
4: -223.3089142, 331.3695374, -254.4911957, 378.2940674, -601.6029663, 585.8605347

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2183387, upper bound: 495.2304941
time: 1.54 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -495.2183387, upper bound: 495.2330140
time: 1.60 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -199.5296326, 303.7568054, -198.7401581, 302.1637268, -501.6933289, 502.4969482
1: -222.7413788, 323.7371216, -221.9353333, 322.1886902, -544.9299316, 545.6724243
2: -226.5391846, 319.2399292, -225.6401978, 317.7497559, -544.2889404, 544.8801270
3: -272.3644714, 375.0136108, -271.4887085, 373.1064453, -645.4708862, 646.5023193
4: -248.2408295, 368.9809570, -247.1549683, 367.3784180, -615.6190796, 616.1359253

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -495.1525669, upper bound: 495.2068845
time: 1.85 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -495.1407015, upper bound: 495.1867382
time: 1.54 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 5.98 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.98
Output dim: 0, lower bound: -495.2494817, upper bound: 495.2486410
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.98
Output dim: 0, lower bound: -495.2461071, upper bound: 495.2439295
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.98
Output dim: 0, lower bound: -495.2494817, upper bound: 495.2486410
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.98
Output dim: 0, lower bound: -495.2461071, upper bound: 495.2439295
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.98
Output dim: 0, lower bound: -495.2362467, upper bound: 495.2349294
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.98
Output dim: 0, lower bound: -495.2362467, upper bound: 495.2349294
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.98
Output dim: 0, lower bound: -495.2362467, upper bound: 495.2349294
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.98
Output dim: 0, lower bound: -495.2362467, upper bound: 495.2349294
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.98
Output dim: 0, lower bound: -495.2583622, upper bound: 495.2556043
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.98
Output dim: 0, lower bound: -495.2605428, upper bound: 495.2605428
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.98
Output dim: 0, lower bound: -495.2583622, upper bound: 495.2556043
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.98
Output dim: 0, lower bound: -495.2605428, upper bound: 495.2609945
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.98
Output dim: 0, lower bound: -495.2544628, upper bound: 495.2535120
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.98
Output dim: 0, lower bound: -495.2573253, upper bound: 495.2593665
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.98
Output dim: 0, lower bound: -495.2544628, upper bound: 495.2537136
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.98
Output dim: 0, lower bound: -495.2573253, upper bound: 495.2598026
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.98
Output dim: 0, lower bound: -495.2494301, upper bound: 495.2487624
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.98
Output dim: 0, lower bound: -495.2461035, upper bound: 495.2460788
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.98
Output dim: 0, lower bound: -495.2494301, upper bound: 495.2486410
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.98
Output dim: 0, lower bound: -495.2461035, upper bound: 495.2439295
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.98
Output dim: 0, lower bound: -495.2183387, upper bound: 495.2304941
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.98
Output dim: 0, lower bound: -495.2183387, upper bound: 495.2330140
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.98
Output dim: 0, lower bound: -495.2183387, upper bound: 495.2304941
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.98
Output dim: 0, lower bound: -495.2183387, upper bound: 495.2330140
IS_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.98
Output dim: 0, lower bound: -495.1525669, upper bound: 495.2068845
IS_A1_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.98
Output dim: 0, lower bound: -495.1407015, upper bound: 495.1867382
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.98
Output dim: 0, lower bound: -495.2431245, upper bound: 495.2517639
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.98
Output dim: 0, lower bound: -495.2429454, upper bound: 495.2523544
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.98
Output dim: 0, lower bound: -495.2431245, upper bound: 495.2517639
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.98
Output dim: 0, lower bound: -495.2330654, upper bound: 495.2212443
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.98
Output dim: 0, lower bound: -495.2330654, upper bound: 495.2212443
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.98
Output dim: 0, lower bound: -495.2330654, upper bound: 495.2212443
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.98
Output dim: 0, lower bound: -495.2330654, upper bound: 495.2212443
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.98
Output dim: 0, lower bound: -495.2530667, upper bound: 495.2468101
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.98
Output dim: 0, lower bound: -495.2530667, upper bound: 495.2468101
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.98
Output dim: 0, lower bound: -495.2530667, upper bound: 495.2468101
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.98
Output dim: 0, lower bound: -495.2530667, upper bound: 495.2468101
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.98
Output dim: 0, lower bound: -495.2406128, upper bound: 495.2444009
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.98
Output dim: 0, lower bound: -495.2423252, upper bound: 495.2423252
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.98
Output dim: 0, lower bound: -495.2406128, upper bound: 495.2444009
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.98
Output dim: 0, lower bound: -495.2423252, upper bound: 495.2423252
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=571.2453002929688
rel_dist={0: [-495.2676133325849, 495.2676133325849]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 1090.76 seconds
