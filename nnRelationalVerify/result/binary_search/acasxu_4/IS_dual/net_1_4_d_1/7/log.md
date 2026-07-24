## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_4.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 398.85261092052


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315)
1: (-197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482)
2: (-197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371)
3: (-234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619)
4: (-201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945)

## BASE Result
execution time: IAR + LP analysis = 2.42 + 2.43 = 4.85 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -398.9390144, upper bound: 398.9390144


# Binary Search by BASE starts (time budget: 1195.15 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=482.57733154296875
rel_dist={0: [-398.93901443925324, 398.93901443925324]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=482.57733154296875
rel_dist={0: [-398.9373570313671, 398.9373570313671]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=482.57733154296875
rel_dist={0: [-398.93352538884415, 398.93352538884415]}

## Binary search (step 3) starts
Candidate diff: 0.0625000


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=482.57733154296875
rel_dist={0: [-398.9304526193936, 398.9304526193936]}

## Binary search (step 4) starts
Candidate diff: 0.0312500


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=482.57733154296875
rel_dist={0: [-398.9286294160353, 398.9286294160353]}

## Binary search (step 5) starts
Candidate diff: 0.0156250


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=482.57733154296875
rel_dist={0: [-398.9271246178754, 398.9271246178754]}

## Binary search (step 6) starts
Candidate diff: 0.0078125


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0078125, mid=0.0078125, abs_max=482.57733154296875
rel_dist={0: [-398.92607928857916, 398.9260792885791]}

## Binary search (step 7) starts
Candidate diff: 0.0039062


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0039062, mid=0.0039062, abs_max=482.57733154296875
rel_dist={0: [-398.92552524542134, 398.92552524542134]}

## Binary search (step 8) starts
Candidate diff: 0.0019531


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0019531, mid=0.0019531, abs_max=482.57733154296875
rel_dist={0: [-398.92524179851074, 398.9252417985108]}

## Binary search (step 9) starts
Candidate diff: 0.0009766


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0009766, mid=0.0009766, abs_max=482.57733154296875
rel_dist={0: [-398.92509988427423, 398.92509988427435]}

## Binary search (step 10) starts
Candidate diff: 0.0004883


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0004883, mid=0.0004883, abs_max=482.57733154296875
rel_dist={0: [-398.9250267544338, 398.92502675443393]}

## Binary search (step 11) starts
Candidate diff: 0.0002441


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0002441, mid=0.0002441, abs_max=482.57733154296875
rel_dist={0: [-398.9249900282473, 398.9249900282473]}

## Binary search (step 12) starts
Candidate diff: 0.0001221


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0001221, mid=0.0001221, abs_max=482.57733154296875
rel_dist={0: [-398.9249713707402, 398.9249713707402]}

## Binary search (step 13) starts
Candidate diff: 0.0000610


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000610, mid=0.0000610, abs_max=482.57733154296875
rel_dist={0: [-398.92496201422057, 398.9249620121283]}

## Binary search (step 14) starts
Candidate diff: 0.0000305


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000305, mid=0.0000305, abs_max=482.57733154296875
rel_dist={0: [-398.9249573327561, 398.9249573331433]}

## Binary search (step 15) starts
Candidate diff: 0.0000153


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000153, mid=0.0000153, abs_max=482.57733154296875
rel_dist={0: [-398.9249549942733, 398.9249549942733]}

## Binary search (step 16) starts
Candidate diff: 0.0000076


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000076, mid=0.0000076, abs_max=482.57733154296875
rel_dist={0: [-398.92495382392326, 398.92495382601146]}

## Binary search (step 17) starts
Candidate diff: 0.0000038


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000038, mid=0.0000038, abs_max=482.57733154296875
rel_dist={0: [-398.9249532635105, 398.924953264239]}

## Binary search (step 18) starts
Candidate diff: 0.0000019


## IAR start
Binary search (step 18): status=Status.UNKNOWN, low=0.0000000, high=0.0000019, mid=0.0000019, abs_max=482.57733154296875
rel_dist={0: [-398.9249529703916, 398.92495299969676]}

## Binary search (step 19) starts
Candidate diff: 0.0000010


## IAR start
Binary search (step 19): status=Status.UNKNOWN, low=0.0000000, high=0.0000010, mid=0.0000010, abs_max=482.57733154296875
rel_dist={0: [-398.92495289759415, 398.92495292160686]}

## Binary Search Result
Binary search time: 102.92 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 1092.23 seconds

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8875484, upper bound: 398.9377021
time: 1.01 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8876411, upper bound: 398.8876411
time: 1.02 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.24 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 2.24
Output dim: 0, lower bound: -398.8875484, upper bound: 398.9377021
IS_B2, status: Status.UNKNOWN, split count: 1, time: 2.24
Output dim: 0, lower bound: -398.8876411, upper bound: 398.8876411

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -179.2202911, 303.3570251, -164.8828735, 273.6366882, -452.8569946, 468.2398987
1: -197.4927673, 268.0154114, -181.5148926, 242.5971832, -440.0898438, 449.5303040
2: -197.7517548, 272.0600586, -181.4661713, 246.9006042, -444.6523438, 453.5261230
3: -234.1109924, 308.6250000, -214.6846161, 279.4453430, -513.5563354, 523.3095093
4: -201.8509827, 312.5909424, -184.8080750, 283.3780212, -485.2290039, 497.3990173

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8293573, upper bound: 398.9375198
time: 1.62 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8853268, upper bound: 398.9376444
time: 1.41 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -178.9141998, 302.9039917, -382.1474609, 576.9548950, -751.1802979, 672.2724609
1: -197.1576538, 267.6112671, -417.8582153, 527.3723145, -719.1680908, 673.7185669
2: -197.4189301, 271.6546631, -416.7462769, 536.6198730, -728.6610718, 678.5336914
3: -233.7204895, 308.1575012, -487.7610474, 608.5759277, -836.0215454, 787.2104492
4: -201.5159454, 312.1154175, -418.2799072, 614.7428589, -813.9605713, 722.1043701

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8876411, upper bound: 398.8874032
time: 1.44 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8876411, upper bound: 398.8876411
time: 1.81 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 5.77 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 5.77
Output dim: 0, lower bound: -398.8293573, upper bound: 398.9375198
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 5.77
Output dim: 0, lower bound: -398.8853268, upper bound: 398.9376444
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 5.77
Output dim: 0, lower bound: -398.8876411, upper bound: 398.8874032
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 5.77
Output dim: 0, lower bound: -398.8876411, upper bound: 398.8876411

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -167.6224060, 286.5415649, -164.6194916, 273.2920837, -440.9144592, 451.1609802
1: -184.9529266, 253.3672943, -181.2321930, 242.2859344, -427.2388611, 434.5994873
2: -185.2733612, 256.9786682, -181.1835632, 246.5796051, -431.8528748, 438.1622314
3: -219.7947693, 291.7548828, -214.3645935, 279.0867310, -498.8815002, 506.1194763
4: -189.7115326, 295.0166626, -184.5361481, 283.0044250, -472.7159424, 479.5527954

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8270030, upper bound: 398.9187404
time: 1.04 seconds

## Relational analysis of IS_B1_A1_B2

### Relational analysis result of IS_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8293573, upper bound: 398.9218237
time: 1.07 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -179.0504303, 303.0819092, -164.8828735, 273.6366882, -452.6871338, 467.9647217
1: -197.3061523, 267.7675171, -181.5148926, 242.5971832, -439.9032593, 449.2824097
2: -197.5640411, 271.8102417, -181.4661713, 246.9006042, -444.4646301, 453.2763062
3: -233.8892059, 308.3383789, -214.6846161, 279.4453430, -513.3345337, 523.0228882
4: -201.6599731, 312.3043518, -184.8080750, 283.3780212, -485.0379944, 497.1124268

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B1_A2_A1

### Relational analysis result of IS_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8853268, upper bound: 398.9374238
time: 0.82 seconds

## Relational analysis of IS_B1_A2_A2

### Relational analysis result of IS_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8853268, upper bound: 398.9376444
time: 0.93 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -164.8828735, 273.6366882, -382.0796814, 576.8610229, -737.2622070, 643.0418091
1: -181.5148926, 242.5971832, -417.7807922, 527.2816162, -703.6503296, 648.8524170
2: -181.4661713, 246.9006042, -416.6720581, 536.5310059, -712.7783203, 654.2753906
3: -214.6846161, 279.4453430, -487.6701965, 608.4691162, -817.0795898, 758.4937744
4: -184.8080750, 283.3780212, -418.2040710, 614.6394653, -797.2938843, 693.3995972

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8861327, upper bound: 398.8835252
time: 1.34 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8874032, upper bound: 398.8874032
time: 1.29 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -382.1474609, 576.9548950, -382.1474609, 576.9548950, -937.3775635, 937.3775635
1: -417.8582153, 527.3723145, -417.8582153, 527.3723145, -923.8921509, 923.8921509
2: -416.7462769, 536.6198730, -416.7462769, 536.6198730, -933.7025757, 933.7026367
3: -487.7610474, 608.5759277, -487.7610474, 608.5759277, -1077.0296631, 1077.0296631
4: -418.2799072, 614.7428589, -418.2799072, 614.7428589, -1018.3641968, 1018.3641357

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A2_A1

### Relational analysis result of IS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8835252, upper bound: 398.8869653
time: 1.72 seconds

## Relational analysis of IS_B2_A2_A2

### Relational analysis result of IS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8874032, upper bound: 398.8876411
time: 1.12 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 7.29 seconds
IS_B1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 7.29
Output dim: 0, lower bound: -398.8270030, upper bound: 398.9187404
IS_B1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 7.29
Output dim: 0, lower bound: -398.8293573, upper bound: 398.9218237
IS_B1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 7.29
Output dim: 0, lower bound: -398.8853268, upper bound: 398.9374238
IS_B1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 7.29
Output dim: 0, lower bound: -398.8853268, upper bound: 398.9376444
IS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 7.29
Output dim: 0, lower bound: -398.8861327, upper bound: 398.8835252
IS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 7.29
Output dim: 0, lower bound: -398.8874032, upper bound: 398.8874032
IS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 7.29
Output dim: 0, lower bound: -398.8835252, upper bound: 398.8869653
IS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 7.29
Output dim: 0, lower bound: -398.8874032, upper bound: 398.8876411

## BFS IS instance: IS_B1_A1_B1

### Backsubstitution after applying IS history:
0: -167.6224060, 286.5415649, -147.6117096, 239.4005432, -407.0229187, 434.1532593
1: -184.9529266, 253.3672943, -162.3287201, 213.6265411, -398.5794678, 415.6960144
2: -185.2733612, 256.9786682, -162.1092987, 218.1314850, -403.4048157, 419.0879211
3: -219.7947693, 291.7548828, -191.6298523, 246.1497192, -465.9444885, 483.3847351
4: -189.7115326, 295.0166626, -164.9687347, 250.1293335, -439.8408813, 459.9853821

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B1_A1

### Relational analysis result of IS_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8270030, upper bound: 398.9166285
time: 1.57 seconds

## Relational analysis of IS_B1_A1_B1_A2

### Relational analysis result of IS_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8246360, upper bound: 398.9172108
time: 1.55 seconds

## BFS IS instance: IS_B1_A1_B2

### Backsubstitution after applying IS history:
0: -167.3087311, 286.0294495, -167.0842896, 273.6861572, -440.9948730, 453.1136780
1: -184.6111755, 252.9243927, -184.4223785, 245.0608063, -429.6718140, 437.3466797
2: -184.9279175, 256.5292358, -184.0300598, 249.9877167, -434.9156189, 440.5592957
3: -219.3984985, 291.2437439, -218.7873993, 282.0509033, -501.4494019, 510.0311279
4: -189.3706055, 294.4966431, -188.3018951, 286.2449646, -475.6155701, 482.7985229

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_B2_A1

### Relational analysis result of IS_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8181067, upper bound: 398.8912952
time: 1.34 seconds

## Relational analysis of IS_B1_A1_B2_A2

### Relational analysis result of IS_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8289770, upper bound: 398.8950165
time: 1.05 seconds

## BFS IS instance: IS_B1_A2_A1

### Backsubstitution after applying IS history:
0: -164.7154694, 273.3681946, -164.8828735, 273.6366882, -438.3521423, 438.2510681
1: -181.3314667, 242.3555298, -181.5148926, 242.5971832, -423.9286194, 423.8704224
2: -181.2819214, 246.6568451, -181.4661713, 246.9006042, -428.1825256, 428.1229553
3: -214.4669952, 279.1690674, -214.6846161, 279.4453430, -493.9123535, 493.8536987
4: -184.6207733, 283.0977783, -184.8080750, 283.3780212, -467.9987793, 467.9058228

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_A1_B1

### Relational analysis result of IS_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8838808, upper bound: 398.9074096
time: 1.41 seconds

## Relational analysis of IS_B1_A2_A1_B2

### Relational analysis result of IS_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8853262, upper bound: 398.9365419
time: 1.60 seconds

## BFS IS instance: IS_B1_A2_A2

### Backsubstitution after applying IS history:
0: -381.9302063, 576.5999756, -164.8828735, 273.6366882, -642.8741455, 736.9931030
1: -417.6162109, 527.0520020, -181.5148926, 242.5971832, -648.6702271, 703.4113770
2: -416.5052795, 536.2994995, -181.4661713, 246.9006042, -654.0929565, 712.5368042
3: -487.4714661, 608.2064819, -214.6846161, 279.4453430, -758.2793579, 816.8065796
4: -418.0336304, 614.3737183, -184.8080750, 283.3780212, -693.2185669, 797.0196533

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B1_A2_A2_B1

### Relational analysis result of IS_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8829726, upper bound: 398.9188651
time: 1.56 seconds

## Relational analysis of IS_B1_A2_A2_B2

### Relational analysis result of IS_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8853268, upper bound: 398.9219484
time: 2.10 seconds

## BFS IS instance: IS_B2_A1_B1

### Backsubstitution after applying IS history:
0: -164.5063171, 273.0354004, -382.4497070, 574.0690308, -733.8596802, 642.0270996
1: -181.0984497, 242.0555725, -417.8921204, 524.5161743, -700.1280518, 647.6214600
2: -181.0496216, 246.3449860, -416.9318848, 533.7464600, -709.2064819, 653.0651245
3: -214.1831970, 278.8206787, -487.0470581, 605.4106445, -813.1176147, 756.7706909
4: -184.3865967, 282.7389526, -417.6376953, 611.9953613, -793.9360962, 691.6947632

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 24

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_B1_B1

### Relational analysis result of IS_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9360556, upper bound: 398.8293573
time: 1.67 seconds

## Relational analysis of IS_B2_A1_B1_B2

### Relational analysis result of IS_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9360346, upper bound: 398.8817943
time: 0.99 seconds

## BFS IS instance: IS_B2_A1_B2

### Backsubstitution after applying IS history:
0: -164.8828735, 273.6366882, -380.5021362, 573.7785645, -734.0333862, 641.2723999
1: -181.5148926, 242.5971832, -416.0093689, 524.7647705, -701.0164795, 646.9117432
2: -181.4661713, 246.9006042, -414.9014893, 533.9810791, -710.1103516, 652.3329468
3: -214.6846161, 279.4453430, -485.5191650, 605.6049805, -814.0775757, 756.2244263
4: -184.8080750, 283.3780212, -416.3692322, 611.7148438, -794.2625732, 691.4238892

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_B2_B1

### Relational analysis result of IS_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9366379, upper bound: 398.8269902
time: 1.27 seconds

## Relational analysis of IS_B2_A1_B2_B2

### Relational analysis result of IS_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9367625, upper bound: 398.8853262
time: 1.18 seconds

## BFS IS instance: IS_B2_A2_A1

### Backsubstitution after applying IS history:
0: -382.5759277, 574.2286987, -381.6701050, 576.1536865, -936.1232910, 933.9660034
1: -418.0365601, 524.6745605, -417.3292847, 526.6274414, -922.4776611, 920.3557129
2: -417.0697632, 533.9007568, -416.2169495, 535.8653564, -932.3237305, 930.0955200
3: -487.2150269, 605.5978394, -487.1266785, 607.7182617, -1075.0863037, 1073.0585938
4: -417.7779236, 612.1762695, -417.7350159, 613.8937378, -1016.4845581, 1015.0210571

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_A1_A1

### Relational analysis result of IS_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8294011, upper bound: 398.8846631
time: 1.83 seconds

## Relational analysis of IS_B2_A2_A1_A2

### Relational analysis result of IS_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8816176, upper bound: 398.8846422
time: 1.20 seconds

## BFS IS instance: IS_B2_A2_A2

### Backsubstitution after applying IS history:
0: -380.5844727, 573.8866577, -382.1474609, 576.9548950, -935.6211548, 934.1630249
1: -416.1034241, 524.8707886, -417.8582153, 527.3723145, -921.9660645, 921.2737427
2: -414.9913025, 534.0844116, -416.7462769, 536.6198730, -931.7741699, 931.0490112
3: -485.6289673, 605.7301636, -487.7610474, 608.5759277, -1074.7774658, 1074.0456543
4: -416.4608765, 611.8358154, -418.2799072, 614.7428589, -1016.4030762, 1015.3502808

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_A2_B1

### Relational analysis result of IS_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -398.8267415, upper bound: 398.8269573
time: 1.42 seconds

## Relational analysis of IS_B2_A2_A2_B2

### Relational analysis result of IS_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8851495, upper bound: 398.8853701
time: 1.20 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 5.17 seconds
IS_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 5.17
Output dim: 0, lower bound: -398.8270030, upper bound: 398.9166285
IS_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 5.17
Output dim: 0, lower bound: -398.8246360, upper bound: 398.9172108
IS_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 5.17
Output dim: 0, lower bound: -398.8181067, upper bound: 398.8912952
IS_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 5.17
Output dim: 0, lower bound: -398.8289770, upper bound: 398.8950165
IS_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.17
Output dim: 0, lower bound: -398.8838808, upper bound: 398.9074096
IS_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.17
Output dim: 0, lower bound: -398.8853262, upper bound: 398.9365419
IS_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.17
Output dim: 0, lower bound: -398.8829726, upper bound: 398.9188651
IS_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.17
Output dim: 0, lower bound: -398.8853268, upper bound: 398.9219484
IS_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 5.17
Output dim: 0, lower bound: -398.9360556, upper bound: 398.8293573
IS_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 5.17
Output dim: 0, lower bound: -398.9360346, upper bound: 398.8817943
IS_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 5.17
Output dim: 0, lower bound: -398.9366379, upper bound: 398.8269902
IS_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 5.17
Output dim: 0, lower bound: -398.9367625, upper bound: 398.8853262
IS_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 5.17
Output dim: 0, lower bound: -398.8294011, upper bound: 398.8846631
IS_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 5.17
Output dim: 0, lower bound: -398.8816176, upper bound: 398.8846422
IS_B2_A2_A2_B1, status: Status.VERIFIED, split count: 4, time: 5.17
Output dim: 0, lower bound: -398.8267415, upper bound: 398.8269573
IS_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.17
Output dim: 0, lower bound: -398.8851495, upper bound: 398.8853701

## BFS IS instance: IS_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -167.2558746, 283.8082886, -147.2456055, 238.8120575, -406.0679321, 431.0538940
1: -184.2607574, 251.0018311, -161.9241028, 213.0885468, -397.3493042, 412.9259033
2: -184.6286011, 254.4100037, -161.7015228, 217.5588379, -402.1874084, 416.1115112
3: -218.2185974, 289.0091553, -191.1430969, 245.5305023, -463.7490845, 480.1522522
4: -188.5265198, 292.1801758, -164.5487061, 249.5032806, -438.0297852, 456.7287903

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8265999, upper bound: 398.9132731
time: 0.99 seconds

## Relational analysis of IS_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8220308, upper bound: 398.9151764
time: 0.86 seconds

## Relational analysis of IS_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_A1_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8196748, upper bound: 398.8936230
time: 1.37 seconds

## Relational analysis of IS_B1_A1_B1_A1_B2

### Relational analysis result of IS_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8197560, upper bound: 398.9163605
time: 1.32 seconds

## BFS IS instance: IS_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -165.5459442, 282.7898560, -147.6117096, 239.4005432, -404.9464111, 430.4015503
1: -182.6677551, 250.0389709, -162.3287201, 213.6265411, -396.2942200, 412.3676453
2: -182.9571075, 253.6776581, -162.1092987, 218.1314850, -401.0885925, 415.7869263
3: -217.0921326, 287.9343872, -191.6298523, 246.1497192, -463.2418518, 479.5642395
4: -187.3269958, 291.2691345, -164.9687347, 250.1293335, -437.4562988, 456.2378540

Time for backsubstitution: 2.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_B1_A2_B1

### Relational analysis result of IS_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8242329, upper bound: 398.9138868
time: 1.26 seconds

## Relational analysis of IS_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_A1_B1_A2_B1

### Relational analysis result of IS_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8184335, upper bound: 398.8941962
time: 1.41 seconds

## Relational analysis of IS_B1_A1_B1_A2_B2

### Relational analysis result of IS_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8185147, upper bound: 398.9169336
time: 0.86 seconds

## BFS IS instance: IS_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -163.4855652, 278.5896301, -167.0842896, 273.6861572, -437.1717224, 445.6738281
1: -180.3717346, 246.4948120, -184.4223785, 245.0608063, -425.4324646, 430.9171448
2: -180.6103516, 250.1288910, -184.0300598, 249.9877167, -430.5980530, 434.1589355
3: -214.2609711, 283.8935242, -218.7873993, 282.0509033, -496.3118591, 502.6809082
4: -184.8748474, 287.0588074, -188.3018951, 286.2449646, -471.1197510, 475.3607178

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B2_A1_A1

### Relational analysis result of IS_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8181067, upper bound: 398.8912952
time: 0.99 seconds

## Relational analysis of IS_B1_A1_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B2_A1_B1

### Relational analysis result of IS_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8180762, upper bound: 398.8912646
time: 1.10 seconds

## Relational analysis of IS_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A1_B2_A1_A1

### Relational analysis result of IS_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8010308, upper bound: 398.8896430
time: 0.99 seconds

## Relational analysis of IS_B1_A1_B2_A1_A2

### Relational analysis result of IS_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8173913, upper bound: 398.8898219
time: 1.30 seconds

## BFS IS instance: IS_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -181.2113495, 302.1820984, -166.6928101, 273.0551758, -454.2665100, 468.8749084
1: -199.3214722, 267.7629089, -183.9902039, 244.4913483, -443.8127441, 451.7530518
2: -199.7812195, 271.8881226, -183.5986176, 249.4083557, -449.1894836, 455.4866638
3: -235.6087341, 308.4272156, -218.2780609, 281.3977051, -517.0064087, 526.7052612
4: -203.1607208, 312.3583069, -187.8593750, 285.5720520, -488.7327881, 500.2176819

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B2_A2_A1

### Relational analysis result of IS_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8289770, upper bound: 398.8944027
time: 1.07 seconds

## Relational analysis of IS_B1_A1_B2_A2_A2

### Relational analysis result of IS_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8266099, upper bound: 398.8950165
time: 1.07 seconds

## BFS IS instance: IS_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -164.3396759, 272.7685242, -169.4530640, 276.8140869, -441.1537476, 442.2215576
1: -180.9158783, 241.8154449, -186.2131653, 245.6297760, -426.5456543, 428.0285645
2: -180.8663177, 246.1027069, -186.0870819, 250.0730896, -430.9393616, 432.1897278
3: -213.9667206, 278.5460510, -219.2395325, 283.0472107, -497.0139160, 497.7855835
4: -184.2002716, 282.4603577, -188.7692108, 287.2324219, -471.4326782, 471.2295532

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A2_A1_B1_B1

### Relational analysis result of IS_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8933512, upper bound: 398.9039159
time: 0.92 seconds

## Relational analysis of IS_B1_A2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B1_A2_A1_B1_A1

### Relational analysis result of IS_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9158461, upper bound: 398.8982724
time: 1.00 seconds

## Relational analysis of IS_B1_A2_A1_B1_A2

### Relational analysis result of IS_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9191107, upper bound: 398.9006266
time: 0.79 seconds

## BFS IS instance: IS_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -164.7154694, 273.3681946, -162.7719269, 269.7117004, -434.4271851, 436.1401367
1: -181.3314667, 242.3555298, -179.1849213, 239.1481323, -420.4796143, 421.5404663
2: -181.2819214, 246.6568451, -179.1102448, 243.4538879, -424.7358093, 425.7670593
3: -214.4669952, 279.1690674, -211.9000854, 275.4799500, -489.9469299, 491.0691223
4: -184.6207733, 283.0977783, -182.3677216, 279.4620361, -464.0827637, 465.4653931

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A2_A1_B2_A1

### Relational analysis result of IS_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9325258, upper bound: 398.8949330
time: 0.91 seconds

## Relational analysis of IS_B1_A2_A1_B2_A2

### Relational analysis result of IS_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8949114, upper bound: 398.8949114
time: 1.03 seconds

## BFS IS instance: IS_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -381.7029419, 576.3623047, -147.8710938, 239.7438202, -609.5007324, 720.2010498
1: -417.3562012, 526.8021240, -162.6090240, 213.9384460, -621.4396973, 684.7047729
2: -416.2596741, 536.0601807, -162.3888702, 218.4668427, -626.8897095, 693.5509644
3: -487.1772156, 607.9093018, -191.9472809, 246.5103302, -726.6755371, 794.1969604
4: -417.7861633, 614.0896606, -165.2416687, 250.5057068, -661.3460083, 777.5352173

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A2_A2_B1_B1

### Relational analysis result of IS_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8312185, upper bound: 398.9155254
time: 0.92 seconds

## Relational analysis of IS_B1_A2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_A2_B1_A1

### Relational analysis result of IS_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8794401, upper bound: 398.9166075
time: 1.22 seconds

## Relational analysis of IS_B1_A2_A2_B1_A2

### Relational analysis result of IS_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8829719, upper bound: 398.9173354
time: 1.13 seconds

## BFS IS instance: IS_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -381.5826721, 575.9938354, -171.9664307, 279.6744690, -649.6077881, 743.4708862
1: -417.2320862, 526.5345459, -189.5056915, 250.4682312, -656.5818481, 710.7290649
2: -416.1159058, 535.7730103, -189.2267456, 255.7161255, -662.4260864, 719.7545166
3: -487.0131531, 607.6068726, -224.2818756, 288.2797241, -767.3125000, 825.4403687
4: -417.6422119, 613.7663574, -193.1625366, 292.8448486, -702.6820679, 804.2913818

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A2_A2_B2_B1

### Relational analysis result of IS_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8336173, upper bound: 398.9187842
time: 2.13 seconds

## Relational analysis of IS_B1_A2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_A2_B2_A1

### Relational analysis result of IS_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8817943, upper bound: 398.9198720
time: 0.85 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2

### Relational analysis result of IS_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8853262, upper bound: 398.9205999
time: 1.03 seconds

## BFS IS instance: IS_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -164.2421875, 272.6903076, -368.9891663, 553.5608521, -713.3517456, 628.6613770
1: -180.8149567, 241.7438812, -403.3459473, 506.0614929, -681.5751953, 633.1111450
2: -180.7663574, 246.0234222, -402.3356323, 514.7062378, -690.0924072, 638.4816284
3: -213.8624420, 278.4615784, -470.3109436, 584.0073853, -791.7639771, 739.9323120
4: -184.1140137, 282.3647461, -403.1646118, 590.5405884, -772.3822021, 677.1417847

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A1_B1_B1_A1

### Relational analysis result of IS_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9166285, upper bound: 398.8270030
time: 1.24 seconds

## Relational analysis of IS_B2_A1_B1_B1_A2

### Relational analysis result of IS_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9198930, upper bound: 398.8293573
time: 0.91 seconds

## BFS IS instance: IS_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -164.5063171, 273.0354004, -382.3021240, 573.8099365, -733.5924072, 641.8630981
1: -181.0984497, 242.0555725, -417.7292480, 524.2863159, -699.8879395, 647.4426880
2: -181.0496216, 246.3449860, -416.7674561, 533.5149536, -708.9636841, 652.8866577
3: -214.1831970, 278.8206787, -486.8500061, 605.1476440, -812.8430176, 756.5590820
4: -184.3865967, 282.7389526, -417.4686584, 611.7307739, -793.6635742, 691.5146484

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 3

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A1_B1_B2_A1

### Relational analysis result of IS_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9335564, upper bound: 398.8336173
time: 0.86 seconds

## Relational analysis of IS_B2_A1_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_B1_B2_A1

### Relational analysis result of IS_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9319570, upper bound: 398.8703789
time: 0.99 seconds

## Relational analysis of IS_B2_A1_B1_B2_A2

### Relational analysis result of IS_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8943425, upper bound: 398.8703572
time: 1.07 seconds

## BFS IS instance: IS_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -164.6194916, 273.2920837, -365.9750061, 551.0241089, -711.2507324, 626.7259521
1: -181.2321930, 242.2859344, -400.2618408, 504.3094482, -680.4262085, 631.0813599
2: -181.1835632, 246.5796051, -399.1010132, 512.9177856, -688.9387817, 636.4439087
3: -214.3645935, 279.0867310, -467.2979431, 581.9357910, -790.4089966, 737.9288940
4: -184.5361481, 283.0044250, -400.5753479, 588.0422974, -770.4901123, 675.5595093

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A1_B2_B1_A1

### Relational analysis result of IS_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9172108, upper bound: 398.8246360
time: 1.15 seconds

## Relational analysis of IS_B2_A1_B2_B1_A2

### Relational analysis result of IS_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9204753, upper bound: 398.8269902
time: 0.87 seconds

## BFS IS instance: IS_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -164.8828735, 273.6366882, -380.3511353, 573.5148926, -733.7616577, 641.1032715
1: -181.5148926, 242.5971832, -415.8430176, 524.5327148, -700.7751465, 646.7279053
2: -181.4661713, 246.9006042, -414.7329712, 533.7470703, -709.8661499, 652.1486206
3: -214.6846161, 279.4453430, -485.3183899, 605.3397217, -813.8016968, 756.0080566
4: -184.8080750, 283.3780212, -416.1970215, 611.4463501, -793.9854736, 691.2409668

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A1_B2_B2_A1

### Relational analysis result of IS_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9173354, upper bound: 398.8829719
time: 1.09 seconds

## Relational analysis of IS_B2_A1_B2_B2_A2

### Relational analysis result of IS_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9205999, upper bound: 398.8853262
time: 1.00 seconds

## BFS IS instance: IS_B2_A2_A1_A1

### Backsubstitution after applying IS history:
0: -369.1825562, 553.7864990, -381.3941040, 575.7433472, -922.4193115, 913.2495728
1: -403.5670471, 506.2905884, -417.0292969, 526.2485962, -907.6595459, 901.6313477
2: -402.5456543, 514.9274292, -415.9173279, 535.4774780, -917.4364014, 910.8025513
3: -470.5653992, 584.2786255, -486.7786865, 607.2795410, -1057.8895264, 1051.5938721
4: -403.3775024, 590.8005981, -417.4372864, 613.4523926, -1001.8077393, 993.2948608

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A2_A1_A1_A1

### Relational analysis result of IS_B2_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8183040, upper bound: 398.8799966
time: 2.94 seconds

## Relational analysis of IS_B2_A2_A1_A1_A2

### Relational analysis result of IS_B2_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8291743, upper bound: 398.8831042
time: 1.35 seconds

## BFS IS instance: IS_B2_A2_A1_A2

### Backsubstitution after applying IS history:
0: -382.4596558, 574.0285645, -381.6701050, 576.1536865, -935.9897461, 933.7548218
1: -417.9093018, 524.4977417, -417.3292847, 526.6274414, -922.3335571, 920.1665039
2: -416.9402771, 533.7221069, -416.2169495, 535.8653564, -932.1796265, 929.9041748
3: -487.0619507, 605.3965454, -487.1266785, 607.7182617, -1074.9183350, 1072.8431396
4: -417.6451416, 611.9722900, -417.7350159, 613.8937378, -1016.3398438, 1014.8081665

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A2_A1_A2_A1

### Relational analysis result of IS_B2_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8802437, upper bound: 398.8824066
time: 1.27 seconds

## Relational analysis of IS_B2_A2_A1_A2_A2

### Relational analysis result of IS_B2_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8704604, upper bound: 398.8830413
time: 1.26 seconds

## BFS IS instance: IS_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -380.5844727, 573.8866577, -382.0289001, 576.7522583, -935.4076538, 934.0255127
1: -416.1034241, 524.8707886, -417.7287598, 527.1953125, -921.7777710, 921.1257935
2: -414.9913025, 534.0844116, -416.6140137, 536.4409790, -931.5834961, 930.9003296
3: -485.6289673, 605.7301636, -487.6061401, 608.3746338, -1074.5633545, 1073.8745117
4: -416.4608765, 611.8358154, -418.1456299, 614.5377197, -1016.1881714, 1015.2045288

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A2_A2_B2_B1

### Relational analysis result of IS_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8829705, upper bound: 398.8834642
time: 1.48 seconds

## Relational analysis of IS_B2_A2_A2_B2_B2

### Relational analysis result of IS_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8835153, upper bound: 398.8838074
time: 1.41 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 6.50 seconds
IS_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 6.50
Output dim: 0, lower bound: -398.8196748, upper bound: 398.8936230
IS_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 6.50
Output dim: 0, lower bound: -398.8197560, upper bound: 398.9163605
IS_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 6.50
Output dim: 0, lower bound: -398.8184335, upper bound: 398.8941962
IS_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 6.50
Output dim: 0, lower bound: -398.8185147, upper bound: 398.9169336
IS_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 6.50
Output dim: 0, lower bound: -398.8010308, upper bound: 398.8896430
IS_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 6.50
Output dim: 0, lower bound: -398.8173913, upper bound: 398.8898219
IS_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 6.50
Output dim: 0, lower bound: -398.8289770, upper bound: 398.8944027
IS_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 6.50
Output dim: 0, lower bound: -398.8266099, upper bound: 398.8950165
IS_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.50
Output dim: 0, lower bound: -398.9158461, upper bound: 398.8982724
IS_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.50
Output dim: 0, lower bound: -398.9191107, upper bound: 398.9006266
IS_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.50
Output dim: 0, lower bound: -398.9325258, upper bound: 398.8949330
IS_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.50
Output dim: 0, lower bound: -398.8949114, upper bound: 398.8949114
IS_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.50
Output dim: 0, lower bound: -398.8794401, upper bound: 398.9166075
IS_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.50
Output dim: 0, lower bound: -398.8829719, upper bound: 398.9173354
IS_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.50
Output dim: 0, lower bound: -398.8817943, upper bound: 398.9198720
IS_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.50
Output dim: 0, lower bound: -398.8853262, upper bound: 398.9205999
IS_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.50
Output dim: 0, lower bound: -398.9166285, upper bound: 398.8270030
IS_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.50
Output dim: 0, lower bound: -398.9198930, upper bound: 398.8293573
IS_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.50
Output dim: 0, lower bound: -398.9319570, upper bound: 398.8703789
IS_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.50
Output dim: 0, lower bound: -398.8943425, upper bound: 398.8703572
IS_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.50
Output dim: 0, lower bound: -398.9172108, upper bound: 398.8246360
IS_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.50
Output dim: 0, lower bound: -398.9204753, upper bound: 398.8269902
IS_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.50
Output dim: 0, lower bound: -398.9173354, upper bound: 398.8829719
IS_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.50
Output dim: 0, lower bound: -398.9205999, upper bound: 398.8853262
IS_B2_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 6.50
Output dim: 0, lower bound: -398.8183040, upper bound: 398.8799966
IS_B2_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 6.50
Output dim: 0, lower bound: -398.8291743, upper bound: 398.8831042
IS_B2_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 6.50
Output dim: 0, lower bound: -398.8802437, upper bound: 398.8824066
IS_B2_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 6.50
Output dim: 0, lower bound: -398.8704604, upper bound: 398.8830413
IS_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 6.50
Output dim: 0, lower bound: -398.8829705, upper bound: 398.8834642
IS_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 6.50
Output dim: 0, lower bound: -398.8835153, upper bound: 398.8838074

## BFS IS instance: IS_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -167.2558746, 283.8082886, -136.5780029, 218.0846100, -385.3404541, 420.3862915
1: -184.2607574, 251.0018311, -149.9568939, 194.8309174, -379.0916138, 400.9587402
2: -184.6286011, 254.4100037, -149.7628174, 198.8754578, -383.5040588, 404.1727600
3: -218.2185974, 289.0091553, -176.2098389, 224.4806824, -442.6992798, 465.2189941
4: -188.5265198, 292.1801758, -152.0704803, 228.2355347, -416.7619934, 444.2506409

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_B1_A1_B1_B1

### Relational analysis result of IS_B1_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8192722, upper bound: 398.8899310
time: 1.90 seconds

## Relational analysis of IS_B1_A1_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B1_A1_B1_B1

### Relational analysis result of IS_B1_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8173928, upper bound: 398.8920309
time: 1.07 seconds

## Relational analysis of IS_B1_A1_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_B1_A1_B1_B1

### Relational analysis result of IS_B1_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8196748, upper bound: 398.8919035
time: 0.89 seconds

## Relational analysis of IS_B1_A1_B1_A1_B1_B2

### Relational analysis result of IS_B1_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8195602, upper bound: 398.8936230
time: 1.37 seconds

## BFS IS instance: IS_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -167.2558746, 283.8082886, -145.6446381, 235.9797668, -403.2356567, 429.4529114
1: -184.2607574, 251.0018311, -160.1444092, 210.6248016, -394.8855591, 411.1462402
2: -184.6286011, 254.4100037, -159.9486389, 215.1358948, -399.7644958, 414.3586426
3: -218.2185974, 289.0091553, -189.0344543, 242.6940308, -460.9126282, 478.0435791
4: -188.5265198, 292.1801758, -162.7486877, 246.6454620, -435.1719971, 454.9288025

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_B1_A1_B2_B1

### Relational analysis result of IS_B1_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8193703, upper bound: 398.9131431
time: 1.06 seconds

## Relational analysis of IS_B1_A1_B1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B1_A1_B2_B1

### Relational analysis result of IS_B1_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8174775, upper bound: 398.9147914
time: 1.27 seconds

## Relational analysis of IS_B1_A1_B1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_B1_A1_B2_B1

### Relational analysis result of IS_B1_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8196748, upper bound: 398.9153165
time: 1.15 seconds

## Relational analysis of IS_B1_A1_B1_A1_B2_B2

### Relational analysis result of IS_B1_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8196471, upper bound: 398.9163605
time: 1.27 seconds

## BFS IS instance: IS_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -165.5459442, 282.7898560, -136.9716492, 218.7257538, -384.2716370, 419.7614746
1: -182.6677551, 250.0389709, -150.3910828, 195.4166260, -378.0843811, 400.4299316
2: -182.9571075, 253.6776581, -150.2007751, 199.4834442, -382.4405518, 403.8784180
3: -217.0921326, 287.9343872, -176.7296295, 225.1483002, -442.2404175, 464.6640015
4: -187.3269958, 291.2691345, -152.5198669, 228.9081879, -416.2351685, 443.7889709

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_B1_A2_B1_B1

### Relational analysis result of IS_B1_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8180309, upper bound: 398.8905336
time: 0.87 seconds

## Relational analysis of IS_B1_A1_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_B1_A2_B1_B1

### Relational analysis result of IS_B1_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8184335, upper bound: 398.8924734
time: 1.03 seconds

## Relational analysis of IS_B1_A1_B1_A2_B1_B2

### Relational analysis result of IS_B1_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8183189, upper bound: 398.8941930
time: 6.72 seconds

## BFS IS instance: IS_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -165.5459442, 282.7898560, -146.0098724, 236.5679169, -402.1138611, 428.7997437
1: -182.6677551, 250.0389709, -160.5480194, 211.1624298, -393.8301086, 410.5869751
2: -182.9571075, 253.6776581, -160.3553772, 215.7082214, -398.6652832, 414.0330200
3: -217.0921326, 287.9343872, -189.5196533, 243.3129272, -460.4050598, 477.4540405
4: -187.3269958, 291.2691345, -163.1675415, 247.2706451, -434.5976562, 454.4366760

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_B1_A2_B2_B1

### Relational analysis result of IS_B1_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8181290, upper bound: 398.9137457
time: 0.90 seconds

## Relational analysis of IS_B1_A1_B1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_B1_A2_B2_B1

### Relational analysis result of IS_B1_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8185147, upper bound: 398.9158865
time: 1.11 seconds

## Relational analysis of IS_B1_A1_B1_A2_B2_B2

### Relational analysis result of IS_B1_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8184058, upper bound: 398.9169304
time: 1.22 seconds

## BFS IS instance: IS_B1_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -152.2966003, 257.0379028, -167.0842896, 273.6861572, -425.9827271, 424.1221313
1: -167.8355560, 227.3621063, -184.4223785, 245.0608063, -412.8962708, 411.7844238
2: -168.0564270, 230.4158325, -184.0300598, 249.9877167, -418.0441284, 414.4458923
3: -198.6361847, 261.8073730, -218.7873993, 282.0509033, -480.6870728, 480.5947571
4: -171.6596832, 264.8417053, -188.3018951, 286.2449646, -457.9046326, 453.1435852

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B1_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B1_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B2_A1_A1_A1

### Relational analysis result of IS_B1_A1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8010308, upper bound: 398.8896430
time: 1.41 seconds

## Relational analysis of IS_B1_A1_B2_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B2_A1_A1_B1

### Relational analysis result of IS_B1_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8010016, upper bound: 398.8896430
time: 0.95 seconds

## Relational analysis of IS_B1_A1_B2_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_B1_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_B2_A1_A1_B1

### Relational analysis result of IS_B1_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8010308, upper bound: 398.8896430
time: 0.95 seconds

## Relational analysis of IS_B1_A1_B2_A1_A1_B2

### Relational analysis result of IS_B1_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8010015, upper bound: 398.8861227
time: 1.24 seconds

## BFS IS instance: IS_B1_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -161.9835510, 275.8509827, -167.0842896, 273.6861572, -435.6697083, 442.9352417
1: -178.7003479, 244.0861359, -184.4223785, 245.0608063, -423.7611389, 428.5084534
2: -178.9407806, 247.7488403, -184.0300598, 249.9877167, -428.9284058, 431.7789001
3: -212.2783508, 281.1389465, -218.7873993, 282.0509033, -494.3292542, 499.9263306
4: -183.1352386, 284.3349609, -188.3018951, 286.2449646, -469.3801575, 472.6368408

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B1_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B1_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B2_A1_A2_A1

### Relational analysis result of IS_B1_A1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8173913, upper bound: 398.8898219
time: 1.03 seconds

## Relational analysis of IS_B1_A1_B2_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B2_A1_A2_B1

### Relational analysis result of IS_B1_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8173651, upper bound: 398.8897913
time: 1.13 seconds

## Relational analysis of IS_B1_A1_B2_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_B1_A1_B2_A1_A2_A1

### Relational analysis result of IS_B1_A1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.7850851, upper bound: 398.8741768
time: 1.09 seconds

## Relational analysis of IS_B1_A1_B2_A1_A2_A2

### Relational analysis result of IS_B1_A1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8172148, upper bound: 398.8770825
time: 1.32 seconds

## BFS IS instance: IS_B1_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -177.6283264, 295.4164734, -166.2791901, 272.4150696, -450.0433960, 461.6956482
1: -195.2374268, 261.7050171, -183.5349731, 243.8840790, -439.1215210, 445.2399902
2: -195.7688141, 265.6231689, -183.1415710, 248.7853088, -444.5539551, 448.7646484
3: -230.3616180, 301.4996948, -217.7350006, 280.7037659, -511.0653687, 519.2346802
4: -198.7135315, 305.2334290, -187.3905640, 284.8706665, -483.5841980, 492.6239624

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B1_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B1_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B2_A2_A1_B1

### Relational analysis result of IS_B1_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8215569, upper bound: 398.8929755
time: 1.05 seconds

## Relational analysis of IS_B1_A1_B2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A1_B2_A2_A1_A1

### Relational analysis result of IS_B1_A1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8143432, upper bound: 398.8927872
time: 1.62 seconds

## Relational analysis of IS_B1_A1_B2_A2_A1_A2

### Relational analysis result of IS_B1_A1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8217301, upper bound: 398.8928592
time: 1.63 seconds

## BFS IS instance: IS_B1_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -179.4576569, 298.9237976, -166.6928101, 273.0551758, -452.5128174, 465.6166077
1: -197.3815002, 264.8722534, -183.9902039, 244.4913483, -441.8727722, 448.8624573
2: -197.8155670, 268.9839172, -183.5986176, 249.4083557, -447.2239075, 452.5825195
3: -233.2727203, 305.0781860, -218.2780609, 281.3977051, -514.6703491, 523.3562622
4: -201.1079865, 309.0753174, -187.8593750, 285.5720520, -486.6800537, 496.9346924

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B1_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B1_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A1_B2_A2_A2_A1

### Relational analysis result of IS_B1_A1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8134909, upper bound: 398.8934322
time: 1.10 seconds

## Relational analysis of IS_B1_A1_B2_A2_A2_A2

### Relational analysis result of IS_B1_A1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8204888, upper bound: 398.8934618
time: 1.33 seconds

## BFS IS instance: IS_B1_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -147.3500519, 238.9152832, -169.4530640, 276.8140869, -424.1641235, 408.3683472
1: -162.0350647, 213.1843719, -186.2131653, 245.6297760, -407.6647949, 399.3975220
2: -161.8114014, 217.6774597, -186.0870819, 250.0730896, -411.8844910, 403.7645264
3: -191.2624512, 245.6421814, -219.2395325, 283.0472107, -474.3096619, 464.8817139
4: -164.6512909, 249.6275482, -188.7692108, 287.2324219, -451.8837280, 438.3967285

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_A1_B1_A1_A1

### Relational analysis result of IS_B1_A2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8928732, upper bound: 398.8973132
time: 1.41 seconds

## Relational analysis of IS_B1_A2_A1_B1_A1_A2

### Relational analysis result of IS_B1_A2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9156106, upper bound: 398.8973944
time: 1.41 seconds

## BFS IS instance: IS_B1_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -171.4234619, 278.8148499, -169.0765381, 276.2235413, -447.6470032, 447.8913879
1: -188.9084930, 249.6674500, -185.8020325, 245.1124268, -434.0209351, 435.4694824
2: -188.6247711, 254.9014435, -185.6746368, 249.5489502, -438.1737061, 440.5760498
3: -223.5716400, 287.3598022, -218.7618713, 282.4500427, -506.0216675, 506.1216736
4: -192.5501251, 291.9229736, -188.3584442, 286.6248474, -479.1749573, 480.2814331

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A2_A1_B1_A2_B1

### Relational analysis result of IS_B1_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8933485, upper bound: 398.8975370
time: 1.39 seconds

## Relational analysis of IS_B1_A2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_A1_B1_A2_A1

### Relational analysis result of IS_B1_A2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8917618, upper bound: 398.8996578
time: 1.59 seconds

## Relational analysis of IS_B1_A2_A1_B1_A2_A2

### Relational analysis result of IS_B1_A2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9190857, upper bound: 398.8997534
time: 1.28 seconds

## BFS IS instance: IS_B1_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -160.7124023, 265.6525269, -162.7719269, 269.7117004, -430.4241028, 428.4244385
1: -176.8868408, 235.6975098, -179.1849213, 239.1481323, -416.0349426, 414.8823853
2: -176.7672577, 240.0148468, -179.1102448, 243.4538879, -420.2211304, 419.1250916
3: -209.0996704, 271.5521240, -211.9000854, 275.4799500, -484.5795898, 483.4521484
4: -179.9243774, 275.3947449, -182.3677216, 279.4620361, -459.3863525, 457.7623291

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_A2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_A1_B2_A1_A1

### Relational analysis result of IS_B1_A2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9007652, upper bound: 398.8933046
time: 1.28 seconds

## Relational analysis of IS_B1_A2_A1_B2_A1_A2

### Relational analysis result of IS_B1_A2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9317597, upper bound: 398.8934339
time: 1.37 seconds

## BFS IS instance: IS_B1_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -182.4221802, 294.4898682, -162.4573822, 269.1465454, -451.5687256, 456.9472656
1: -200.1904144, 261.8697815, -178.8359528, 238.6531830, -438.8435364, 440.7057495
2: -200.3633575, 266.7596130, -178.7597656, 242.9625549, -443.3259277, 445.5193176
3: -235.5183411, 301.8487244, -211.4857788, 274.9126282, -510.4309692, 513.3344727
4: -202.7409973, 306.6279602, -182.0022278, 278.8891296, -481.6301270, 488.6301880

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B1_A2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B1_A2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_A2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_A1_B2_A2_A1

### Relational analysis result of IS_B1_A2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8691824, upper bound: 398.8933020
time: 1.25 seconds

## Relational analysis of IS_B1_A2_A1_B2_A2_A2

### Relational analysis result of IS_B1_A2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8933869, upper bound: 398.8933870
time: 1.23 seconds

## BFS IS instance: IS_B1_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -382.0689697, 573.5668335, -147.5059204, 239.1561127, -608.4517822, 716.8167725
1: -417.4624329, 524.0303345, -162.2053680, 213.4012756, -620.1903687, 681.1994629
2: -416.5150452, 533.2698364, -161.9820862, 217.8952179, -625.6669922, 689.9918213
3: -486.5473633, 604.8432617, -191.4616547, 245.8920288, -724.9188843, 790.2467651
4: -417.2144775, 611.4401245, -164.8225708, 249.8807220, -659.6296997, 774.1857300

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A2_A2_B1_A1_B1

### Relational analysis result of IS_B1_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8312185, upper bound: 398.9140450
time: 1.86 seconds

## Relational analysis of IS_B1_A2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A2_A2_B1_A1_B1

### Relational analysis result of IS_B1_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8679801, upper bound: 398.9132102
time: 1.54 seconds

## Relational analysis of IS_B1_A2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_A2_B1_A1_A1

### Relational analysis result of IS_B1_A2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8595124, upper bound: 398.9162464
time: 1.28 seconds

## Relational analysis of IS_B1_A2_A2_B1_A1_A2

### Relational analysis result of IS_B1_A2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8787818, upper bound: 398.9163428
time: 1.29 seconds

## BFS IS instance: IS_B1_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -380.1243896, 573.2780762, -147.8710938, 239.7438202, -607.7305908, 716.9702759
1: -415.5836792, 524.2838135, -162.6090240, 213.9384460, -619.4977417, 682.0695801
2: -414.4879761, 533.5086060, -162.3888702, 218.4668427, -624.9461060, 690.8811035
3: -485.0246887, 605.0435791, -191.9472809, 246.5103302, -724.4044800, 791.1931763
4: -415.9501953, 611.1632690, -165.2416687, 250.5057068, -659.3691406, 774.5020142

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A2_A2_B1_A2_B1

### Relational analysis result of IS_B1_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8812331, upper bound: 398.9139763
time: 1.40 seconds

## Relational analysis of IS_B1_A2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B1_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_A2_B1_A2_A1

### Relational analysis result of IS_B1_A2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8612400, upper bound: 398.9170380
time: 1.27 seconds

## Relational analysis of IS_B1_A2_A2_B1_A2_A2

### Relational analysis result of IS_B1_A2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8817671, upper bound: 398.9170862
time: 1.19 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -381.9094543, 573.1038208, -171.5646057, 279.0502625, -648.4898071, 739.9704590
1: -417.2937012, 523.6951294, -189.0636139, 249.8730011, -655.2506714, 707.1353760
2: -416.3273621, 532.9124146, -188.7804871, 255.1091003, -661.1320190, 716.0886230
3: -486.3284607, 604.4648438, -223.7553864, 287.5984802, -765.4583740, 821.3806763
4: -417.0230103, 611.0334473, -192.7070465, 292.1623840, -700.8826904, 800.8381958

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A2_A2_B2_A1_B1

### Relational analysis result of IS_B1_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8336173, upper bound: 398.9174358
time: 1.65 seconds

## Relational analysis of IS_B1_A2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A2_A2_B2_A1_B1

### Relational analysis result of IS_B1_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8703789, upper bound: 398.9164959
time: 1.27 seconds

## Relational analysis of IS_B1_A2_A2_B2_A1_B2

### Relational analysis result of IS_B1_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8703572, upper bound: 398.8943399
time: 1.38 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -380.0058594, 572.9113770, -171.9664307, 279.6744690, -647.8386841, 740.2419434
1: -415.4612427, 524.0180664, -189.5056915, 250.4682312, -654.6411743, 708.0953979
2: -414.3459167, 533.2232666, -189.2267456, 255.7161255, -660.4837036, 717.0864258
3: -484.8626404, 604.7433472, -224.2818756, 288.2797241, -765.0433960, 822.4388428
4: -415.8078918, 610.8419800, -193.1625366, 292.8448486, -700.7063599, 801.2600708

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A2_A2_B2_A2_B1

### Relational analysis result of IS_B1_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8836318, upper bound: 398.9172620
time: 1.45 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2_B2

### Relational analysis result of IS_B1_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8836101, upper bound: 398.8951060
time: 1.26 seconds

## BFS IS instance: IS_B2_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -147.2456055, 238.8120575, -368.7138062, 553.2770996, -696.5187988, 595.1991577
1: -161.9241028, 213.0885468, -403.0308533, 505.7618103, -662.8255615, 605.7770386
2: -161.7015228, 217.5588379, -402.0379028, 514.4197388, -671.0652466, 611.1759644
3: -191.1430969, 245.5305023, -469.9539185, 583.6509399, -769.0951538, 708.2041016
4: -164.5487061, 249.5032806, -402.8649597, 590.2007446, -752.8202515, 645.2024536

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A1_B1_B1_A1_A1

### Relational analysis result of IS_B2_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9151764, upper bound: 398.8220308
time: 0.91 seconds

## Relational analysis of IS_B2_A1_B1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_B1_B1_A1_A1

### Relational analysis result of IS_B2_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9132731, upper bound: 398.8265999
time: 1.03 seconds

## Relational analysis of IS_B2_A1_B1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B2_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_B1_B1_A1_B1

### Relational analysis result of IS_B2_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9162733, upper bound: 398.8123691
time: 1.20 seconds

## Relational analysis of IS_B2_A1_B1_B1_A1_B2

### Relational analysis result of IS_B2_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9163605, upper bound: 398.8197560
time: 0.96 seconds

## BFS IS instance: IS_B2_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -166.6719208, 273.0473633, -368.5831909, 552.8481445, -715.1845093, 629.6119995
1: -183.9684143, 244.4548798, -402.8955383, 505.4563599, -684.0386353, 635.7996216
2: -183.5743103, 249.3661041, -401.8806763, 514.0900879, -692.3266602, 641.3493042
3: -218.2457275, 281.3585815, -469.7722168, 583.3070068, -795.0507202, 742.9048462
4: -187.8344574, 285.5452271, -402.7040405, 589.8272705, -774.9998779, 680.2488403

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A1_B1_B1_A2_A1

### Relational analysis result of IS_B2_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9185672, upper bound: 398.8244296
time: 0.95 seconds

## Relational analysis of IS_B2_A1_B1_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_B1_B1_A2_A1

### Relational analysis result of IS_B2_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9165588, upper bound: 398.8289987
time: 0.97 seconds

## Relational analysis of IS_B2_A1_B1_B1_A2_A2

### Relational analysis result of IS_B2_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8944027, upper bound: 398.8289770
time: 1.15 seconds

## BFS IS instance: IS_B2_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -160.5099945, 265.3304443, -382.2406006, 573.7312012, -729.6176147, 634.0473633
1: -176.6607971, 235.4072876, -417.6589966, 524.2085571, -695.4428101, 640.7909546
2: -176.5422974, 239.7059937, -416.7004089, 533.4391479, -704.4468384, 646.2328491
3: -208.8241119, 271.2149048, -486.7682495, 605.0557251, -807.4407349, 748.8095093
4: -179.6979675, 275.0475159, -417.4003601, 611.6421509, -788.9262695, 683.7250366

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A1_B1_B2_A1_A1

### Relational analysis result of IS_B2_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9292061, upper bound: 398.8223833
time: 1.08 seconds

## Relational analysis of IS_B2_A1_B1_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A1_B1_B2_A1_A1

### Relational analysis result of IS_B2_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9132102, upper bound: 398.8679801
time: 1.11 seconds

## Relational analysis of IS_B2_A1_B1_B2_A1_A2

### Relational analysis result of IS_B2_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9164959, upper bound: 398.8703789
time: 1.10 seconds

## BFS IS instance: IS_B2_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -182.1048737, 294.0213623, -382.0454407, 573.3036499, -750.3108521, 662.2856445
1: -199.8419037, 261.4412842, -417.4490356, 523.8796387, -717.8890381, 667.0769653
2: -200.0145569, 266.3258972, -416.4795837, 533.0982056, -727.2830811, 672.9741821
3: -235.1074677, 301.3526001, -486.5125122, 604.6891479, -833.0995483, 779.1934204
4: -202.3819275, 306.1269531, -417.1725159, 611.2473755, -810.8479614, 714.6563110

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A1_B1_B2_A2_A1

### Relational analysis result of IS_B2_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8916156, upper bound: 398.8223717
time: 1.07 seconds

## Relational analysis of IS_B2_A1_B1_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B2_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_B1_B2_A2_B1

### Relational analysis result of IS_B2_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8927156, upper bound: 398.8528170
time: 0.99 seconds

## Relational analysis of IS_B2_A1_B1_B2_A2_B2

### Relational analysis result of IS_B2_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8927981, upper bound: 398.8654674
time: 1.04 seconds

## BFS IS instance: IS_B2_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -147.6117096, 239.4005432, -365.7072144, 550.7474976, -694.4036865, 593.3043823
1: -162.3287201, 213.6265411, -399.9573975, 504.0173645, -661.6606445, 603.7730103
2: -162.1092987, 218.1314850, -398.8115845, 512.6387329, -669.9001465, 609.1576538
3: -191.6298523, 246.1497192, -466.9510193, 581.5882568, -767.7290039, 706.2384033
4: -164.9687347, 250.1293335, -400.2834778, 587.7111816, -750.9243164, 643.6343384

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_B2_B1_A1_A1

### Relational analysis result of IS_B2_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9138868, upper bound: 398.8242329
time: 1.42 seconds

## Relational analysis of IS_B2_A1_B2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_B2_B1_A1_B1

### Relational analysis result of IS_B2_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9169013, upper bound: 398.8115168
time: 1.17 seconds

## Relational analysis of IS_B2_A1_B2_B1_A1_B2

### Relational analysis result of IS_B2_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9169336, upper bound: 398.8185147
time: 1.43 seconds

## BFS IS instance: IS_B2_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -167.0842896, 273.6861572, -365.6077271, 550.3916626, -713.2037964, 627.7808228
1: -184.4223785, 245.0608063, -399.8558350, 503.7641907, -682.9833984, 633.8726807
2: -184.0300598, 249.9877167, -398.6894531, 512.3641968, -691.2758789, 639.4096069
3: -218.7873993, 282.0509033, -466.8131409, 581.3035278, -793.8016968, 741.0273438
4: -188.3018951, 286.2449646, -400.1614380, 587.4031982, -773.2200928, 678.7689819

Time for backsubstitution: 2.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_B2_B1_A2_A1

### Relational analysis result of IS_B2_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9171725, upper bound: 398.8266316
time: 0.90 seconds

## Relational analysis of IS_B2_A1_B2_B1_A2_A2

### Relational analysis result of IS_B2_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8950165, upper bound: 398.8266099
time: 0.87 seconds

## BFS IS instance: IS_B2_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -147.8710938, 239.7438202, -380.1243896, 573.2780762, -716.9702148, 607.7305908
1: -162.6090240, 213.9384460, -415.5836792, 524.2838135, -682.0695190, 619.4976807
2: -162.3888702, 218.4668427, -414.4879761, 533.5086060, -690.8811035, 624.9461060
3: -191.9472809, 246.5103302, -485.0246887, 605.0435791, -791.1931763, 724.4044800
4: -165.2416687, 250.5057068, -415.9501953, 611.1632690, -774.5019531, 659.3692017

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_B2_B2_A1_A1

### Relational analysis result of IS_B2_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9139763, upper bound: 398.8812331
time: 0.97 seconds

## Relational analysis of IS_B2_A1_B2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_B2_B2_A1_B1

### Relational analysis result of IS_B2_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9170380, upper bound: 398.8612400
time: 0.94 seconds

## Relational analysis of IS_B2_A1_B2_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9170862, upper bound: 398.8817671
time: 1.17 seconds

## BFS IS instance: IS_B2_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -171.9664307, 279.6744690, -380.0058594, 572.9113770, -740.2419434, 647.8386230
1: -189.5056915, 250.4682312, -415.4612427, 524.0180664, -708.0953979, 654.6411743
2: -189.2267456, 255.7161255, -414.3459167, 533.2232666, -717.0864258, 660.4837036
3: -224.2818756, 288.2797241, -484.8626404, 604.7433472, -822.4388428, 765.0434570
4: -193.1625366, 292.8448486, -415.8078918, 610.8419800, -801.2600708, 700.7063599

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_B2_B2_A2_A1

### Relational analysis result of IS_B2_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9172620, upper bound: 398.8836318
time: 1.02 seconds

## Relational analysis of IS_B2_A1_B2_B2_A2_A2

### Relational analysis result of IS_B2_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8951060, upper bound: 398.8836101
time: 1.03 seconds

## BFS IS instance: IS_B2_A2_A1_A1_A1

### Backsubstitution after applying IS history:
0: -366.4680786, 548.3305054, -381.3941040, 575.7433472, -919.3872681, 907.4769287
1: -400.5328064, 501.8377686, -417.0292969, 526.2485962, -904.3294678, 896.9014282
2: -399.4786682, 510.4559631, -415.9173279, 535.4774780, -914.0842285, 906.0599976
3: -466.8620911, 579.2196045, -486.7786865, 607.2795410, -1053.9440918, 1046.2166748
4: -400.1837158, 585.6448364, -417.4372864, 613.4523926, -998.3861694, 987.9367065

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A2_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B2_A2_A1_A1_A1_B1

### Relational analysis result of IS_B2_A2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8183007, upper bound: 398.8799660
time: 1.00 seconds

## Relational analysis of IS_B2_A2_A1_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A1_A1_A1_A1

### Relational analysis result of IS_B2_A2_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8012214, upper bound: 398.8785156
time: 0.94 seconds

## Relational analysis of IS_B2_A2_A1_A1_A1_A2

### Relational analysis result of IS_B2_A2_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8175819, upper bound: 398.8786944
time: 1.11 seconds

## BFS IS instance: IS_B2_A2_A1_A1_A2

### Backsubstitution after applying IS history:
0: -390.0310974, 581.9844971, -380.9176025, 574.9323120, -939.7401733, 938.5999146
1: -426.0952759, 533.4058228, -416.5041809, 525.5325317, -926.9692383, 926.0060425
2: -425.0762024, 542.6538086, -415.3876343, 534.7572021, -936.8180542, 935.8398438
3: -496.2986755, 615.8410034, -486.1549683, 606.4581909, -1080.8406982, 1079.9276123
4: -425.2524109, 622.2828369, -416.8985596, 612.6248779, -1020.9494629, 1022.7091064

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A2_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B2_A2_A1_A1_A2_B1

### Relational analysis result of IS_B2_A2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8217814, upper bound: 398.8816769
time: 1.02 seconds

## Relational analysis of IS_B2_A2_A1_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A1_A1_A2_A1

### Relational analysis result of IS_B2_A2_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8145338, upper bound: 398.8816598
time: 1.17 seconds

## Relational analysis of IS_B2_A2_A1_A1_A2_A2

### Relational analysis result of IS_B2_A2_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8219207, upper bound: 398.8817318
time: 1.05 seconds

## BFS IS instance: IS_B2_A2_A1_A2_A1

### Backsubstitution after applying IS history:
0: -379.8110046, 568.6788330, -381.6701050, 576.1536865, -933.0338135, 928.1013794
1: -414.9362183, 520.1199341, -417.3292847, 526.6274414, -919.0754395, 915.5214233
2: -413.9431458, 529.3345337, -416.2169495, 535.8653564, -928.9078979, 925.2551880
3: -483.4351807, 600.4346313, -487.1266785, 607.7182617, -1071.0587158, 1067.5688477
4: -414.5108643, 606.9216309, -417.7350159, 613.8937378, -1012.9866943, 1009.5650024

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B2_A2_A1_A2_A1_B1

### Relational analysis result of IS_B2_A2_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8271641, upper bound: 398.8800263
time: 0.90 seconds

## Relational analysis of IS_B2_A2_A1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_A1_A2_A1_A1

### Relational analysis result of IS_B2_A2_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8613280, upper bound: 398.8809387
time: 1.20 seconds

## Relational analysis of IS_B2_A2_A1_A2_A1_A2

### Relational analysis result of IS_B2_A2_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8797586, upper bound: 398.8811011
time: 1.31 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 8.12 seconds
IS_B1_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.8196748, upper bound: 398.8919035
IS_B1_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.8195602, upper bound: 398.8936230
IS_B1_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.8196748, upper bound: 398.9153165
IS_B1_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.8196471, upper bound: 398.9163605
IS_B1_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.8184335, upper bound: 398.8924734
IS_B1_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.8183189, upper bound: 398.8941930
IS_B1_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.8185147, upper bound: 398.9158865
IS_B1_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.8184058, upper bound: 398.9169304
IS_B1_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.8010308, upper bound: 398.8896430
IS_B1_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.8010015, upper bound: 398.8861227
IS_B1_A1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.7850851, upper bound: 398.8741768
IS_B1_A1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.8172148, upper bound: 398.8770825
IS_B1_A1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.8143432, upper bound: 398.8927872
IS_B1_A1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.8217301, upper bound: 398.8928592
IS_B1_A1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.8134909, upper bound: 398.8934322
IS_B1_A1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.8204888, upper bound: 398.8934618
IS_B1_A2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.8928732, upper bound: 398.8973132
IS_B1_A2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.9156106, upper bound: 398.8973944
IS_B1_A2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.8917618, upper bound: 398.8996578
IS_B1_A2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.9190857, upper bound: 398.8997534
IS_B1_A2_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.9007652, upper bound: 398.8933046
IS_B1_A2_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.9317597, upper bound: 398.8934339
IS_B1_A2_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.8691824, upper bound: 398.8933020
IS_B1_A2_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.8933869, upper bound: 398.8933870
IS_B1_A2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.8595124, upper bound: 398.9162464
IS_B1_A2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.8787818, upper bound: 398.9163428
IS_B1_A2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.8612400, upper bound: 398.9170380
IS_B1_A2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.8817671, upper bound: 398.9170862
IS_B1_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.8703789, upper bound: 398.9164959
IS_B1_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.8703572, upper bound: 398.8943399
IS_B1_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.8836318, upper bound: 398.9172620
IS_B1_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.8836101, upper bound: 398.8951060
IS_B2_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.9162733, upper bound: 398.8123691
IS_B2_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.9163605, upper bound: 398.8197560
IS_B2_A1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.9165588, upper bound: 398.8289987
IS_B2_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.8944027, upper bound: 398.8289770
IS_B2_A1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.9132102, upper bound: 398.8679801
IS_B2_A1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.9164959, upper bound: 398.8703789
IS_B2_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.8927156, upper bound: 398.8528170
IS_B2_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.8927981, upper bound: 398.8654674
IS_B2_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.9169013, upper bound: 398.8115168
IS_B2_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.9169336, upper bound: 398.8185147
IS_B2_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.9171725, upper bound: 398.8266316
IS_B2_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.8950165, upper bound: 398.8266099
IS_B2_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.9170380, upper bound: 398.8612400
IS_B2_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.9170862, upper bound: 398.8817671
IS_B2_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.9172620, upper bound: 398.8836318
IS_B2_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.8951060, upper bound: 398.8836101
IS_B2_A2_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.8012214, upper bound: 398.8785156
IS_B2_A2_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.8175819, upper bound: 398.8786944
IS_B2_A2_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.8145338, upper bound: 398.8816598
IS_B2_A2_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.8219207, upper bound: 398.8817318
IS_B2_A2_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.8613280, upper bound: 398.8809387
IS_B2_A2_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 8.12
Output dim: 0, lower bound: -398.8797586, upper bound: 398.8811011
IS_B2_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 8.12
Output dim: 0, lower bound: -398.8704604, upper bound: 398.8830413
IS_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 8.12
Output dim: 0, lower bound: -398.8829705, upper bound: 398.8834642
IS_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 8.12
Output dim: 0, lower bound: -398.8835153, upper bound: 398.8838074
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=482.57733154296875
rel_dist={0: [-398.93901443925324, 398.93901443925324]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8867203, upper bound: 398.9348608
time: 1.08 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8866934, upper bound: 398.8866934
time: 0.91 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.19 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 2.19
Output dim: 0, lower bound: -398.8867203, upper bound: 398.9348608
IS_B2, status: Status.UNKNOWN, split count: 1, time: 2.19
Output dim: 0, lower bound: -398.8866934, upper bound: 398.8866934

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -179.2202911, 303.3570251, -164.8828735, 273.6366882, -452.8569946, 468.2398987
1: -197.4927673, 268.0154114, -181.5148926, 242.5971832, -440.0898438, 449.5303040
2: -197.7517548, 272.0600586, -181.4661713, 246.9006042, -444.6523438, 453.5261230
3: -234.1109924, 308.6250000, -214.6846161, 279.4453430, -513.5563354, 523.3095093
4: -201.8509827, 312.5909424, -184.8080750, 283.3780212, -485.2290039, 497.3990173

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8865091, upper bound: 398.8865091
time: 1.06 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8865091, upper bound: 398.8866934
time: 1.25 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -178.5576172, 302.3097534, -382.0864258, 576.8370972, -750.7131348, 671.6202393
1: -196.7674561, 267.0998230, -417.7886047, 527.2669067, -718.6796265, 673.1426392
2: -197.0205383, 271.1442261, -416.6779175, 536.5142822, -728.1585083, 677.9602661
3: -233.2631989, 307.5670166, -487.6743774, 608.4529419, -835.4491577, 786.5380249
4: -201.1223145, 311.5185852, -418.2083435, 614.6218872, -813.4503174, 721.4415283

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8866934, upper bound: 398.8865091
time: 1.11 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8866934, upper bound: 398.8866934
time: 0.93 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.51 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 4.51
Output dim: 0, lower bound: -398.8865091, upper bound: 398.8865091
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 4.51
Output dim: 0, lower bound: -398.8865091, upper bound: 398.8866934
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 4.51
Output dim: 0, lower bound: -398.8866934, upper bound: 398.8865091
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 4.51
Output dim: 0, lower bound: -398.8866934, upper bound: 398.8866934

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -164.8828735, 273.6366882, -164.8828735, 273.6366882, -438.5195618, 438.5195618
1: -181.5148926, 242.5971832, -181.5148926, 242.5971832, -424.1119995, 424.1119995
2: -181.4661713, 246.9006042, -181.4661713, 246.9006042, -428.3666992, 428.3666992
3: -214.6846161, 279.4453430, -214.6846161, 279.4453430, -494.1299438, 494.1299438
4: -184.8080750, 283.3780212, -184.8080750, 283.3780212, -468.1860962, 468.1860962

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8851440, upper bound: 398.9066869
time: 1.04 seconds

## Relational analysis of IS_B1_A1_B2

### Relational analysis result of IS_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8866066, upper bound: 398.9338079
time: 1.14 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -381.9971313, 576.6977539, -164.8828735, 273.6366882, -642.9606934, 737.1074219
1: -417.6867065, 527.1360474, -181.5148926, 242.5971832, -648.7600098, 703.5108643
2: -416.5794983, 536.3850708, -181.4661713, 246.9006042, -654.1838989, 712.6371460
3: -487.5523071, 608.2994385, -214.6846161, 279.4453430, -758.3770142, 816.9176636
4: -418.1069336, 614.4724121, -184.8080750, 283.3780212, -693.3045044, 797.1302490

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8841369, upper bound: 398.9155525
time: 0.89 seconds

## Relational analysis of IS_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8867203, upper bound: 398.9196829
time: 1.44 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -164.8828735, 273.6366882, -381.9971313, 576.6977539, -737.1074219, 642.9606934
1: -181.5148926, 242.5971832, -417.6867065, 527.1360474, -703.5108643, 648.7600708
2: -181.4661713, 246.9006042, -416.5794983, 536.3850708, -712.6371460, 654.1838989
3: -214.6846161, 279.4453430, -487.5523071, 608.2994385, -816.9176636, 758.3770752
4: -184.8080750, 283.3780212, -418.1069336, 614.4724121, -797.1302490, 693.3044434

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8851440, upper bound: 398.8828823
time: 1.22 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8863927, upper bound: 398.8863927
time: 1.18 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -382.1474609, 576.9548950, -382.1474609, 576.9548950, -937.3775635, 937.3775635
1: -417.8582153, 527.3723145, -417.8582153, 527.3723145, -923.8921509, 923.8921509
2: -416.7462769, 536.6198730, -416.7462769, 536.6198730, -933.7025757, 933.7026367
3: -487.7610474, 608.5759277, -487.7610474, 608.5759277, -1077.0296631, 1077.0296631
4: -418.2799072, 614.7428589, -418.2799072, 614.7428589, -1018.3641968, 1018.3641357

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8850391, upper bound: 398.8663845
time: 1.08 seconds

## Relational analysis of IS_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8850391, upper bound: 398.8852699
time: 1.19 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 6.58 seconds
IS_B1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 6.58
Output dim: 0, lower bound: -398.8851440, upper bound: 398.9066869
IS_B1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 6.58
Output dim: 0, lower bound: -398.8866066, upper bound: 398.9338079
IS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 6.58
Output dim: 0, lower bound: -398.8841369, upper bound: 398.9155525
IS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 6.58
Output dim: 0, lower bound: -398.8867203, upper bound: 398.9196829
IS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 6.58
Output dim: 0, lower bound: -398.8851440, upper bound: 398.8828823
IS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 6.58
Output dim: 0, lower bound: -398.8863927, upper bound: 398.8863927
IS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 6.58
Output dim: 0, lower bound: -398.8850391, upper bound: 398.8663845
IS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 6.58
Output dim: 0, lower bound: -398.8850391, upper bound: 398.8852699

## BFS IS instance: IS_B1_A1_B1

### Backsubstitution after applying IS history:
0: -163.4986877, 271.4302979, -169.4530640, 276.8140869, -440.3127747, 440.8833313
1: -179.9843292, 240.6073303, -186.2131653, 245.6297760, -425.6141052, 426.8204956
2: -179.9349518, 244.8607941, -186.0870819, 250.0730896, -430.0080566, 430.9478760
3: -212.8421326, 277.1510925, -219.2395325, 283.0472107, -495.8893127, 496.3906250
4: -183.2584534, 281.0318298, -188.7692108, 287.2324219, -470.4908752, 469.8010254

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_B1_B1

### Relational analysis result of IS_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8919732, upper bound: 398.9025754
time: 1.18 seconds

## Relational analysis of IS_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_A1_B1_B1

### Relational analysis result of IS_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9326858, upper bound: 398.8834642
time: 0.94 seconds

## Relational analysis of IS_B1_A1_B1_B2

### Relational analysis result of IS_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9327872, upper bound: 398.9064325
time: 0.94 seconds

## BFS IS instance: IS_B1_A1_B2

### Backsubstitution after applying IS history:
0: -164.8828735, 273.6366882, -162.7719269, 269.7117004, -434.5945740, 436.4086304
1: -181.5148926, 242.5971832, -179.1849213, 239.1481323, -420.6630249, 421.7820740
2: -181.4661713, 246.9006042, -179.1102448, 243.4538879, -424.9199829, 426.0108032
3: -214.6846161, 279.4453430, -211.9000854, 275.4799500, -490.1645508, 491.3453979
4: -184.8080750, 283.3780212, -182.3677216, 279.4620361, -464.2701111, 465.7456970

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_B2_B1

### Relational analysis result of IS_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8943235, upper bound: 398.9304202
time: 0.80 seconds

## Relational analysis of IS_B1_A1_B2_B2

### Relational analysis result of IS_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8942847, upper bound: 398.8942847
time: 1.22 seconds

## BFS IS instance: IS_B1_A2_B1

### Backsubstitution after applying IS history:
0: -380.4057922, 573.6955566, -147.8710938, 239.7438202, -608.0903320, 717.4170532
1: -415.8912048, 524.6560669, -162.6090240, 213.9384460, -619.8707275, 682.4618530
2: -414.7967529, 533.8960571, -162.3888702, 218.4668427, -625.3291626, 691.2916870
3: -485.3917542, 605.4765625, -191.9472809, 246.5103302, -724.8074951, 791.6472778
4: -416.2590637, 611.6026611, -165.2416687, 250.5057068, -659.7391357, 774.9636841

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A2_B1_A1

### Relational analysis result of IS_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8269982, upper bound: 398.9101895
time: 1.24 seconds

## Relational analysis of IS_B1_A2_B1_A2

### Relational analysis result of IS_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8823946, upper bound: 398.9154449
time: 0.97 seconds

## BFS IS instance: IS_B1_A2_B2

### Backsubstitution after applying IS history:
0: -379.6451111, 572.6901855, -171.9664307, 279.6744690, -647.6314087, 740.1663818
1: -415.0778198, 523.6565552, -189.5056915, 250.4682312, -654.3932495, 707.8518066
2: -413.9509583, 532.8567505, -189.2267456, 255.7161255, -660.2309570, 716.8424072
3: -484.4406738, 604.2603149, -224.2818756, 288.2797241, -764.7169189, 822.0950928
4: -415.4573975, 610.4019775, -193.1625366, 292.8448486, -700.4754639, 800.9247437

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A2_B2_A1

### Relational analysis result of IS_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8293560, upper bound: 398.9187599
time: 1.50 seconds

## Relational analysis of IS_B1_A2_B2_A2

### Relational analysis result of IS_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8848215, upper bound: 398.9196317
time: 1.07 seconds

## BFS IS instance: IS_B2_A1_B1

### Backsubstitution after applying IS history:
0: -163.4986877, 271.4302979, -382.3299866, 573.8519897, -732.6174316, 640.2775879
1: -179.9843292, 240.6073303, -417.7549744, 524.3188477, -698.7888794, 645.8612671
2: -179.9349518, 244.8607941, -416.7985229, 533.5491333, -707.8853149, 651.2239380
3: -212.8421326, 277.1510925, -486.8782043, 605.1806030, -811.5085449, 754.7659912
4: -183.2584534, 281.0318298, -417.4983215, 611.7680054, -792.5064697, 689.7202759

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_B1_B1

### Relational analysis result of IS_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9310214, upper bound: 398.8293560
time: 1.11 seconds

## Relational analysis of IS_B2_A1_B1_B2

### Relational analysis result of IS_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9333194, upper bound: 398.8814962
time: 0.97 seconds

## BFS IS instance: IS_B2_A1_B2

### Backsubstitution after applying IS history:
0: -164.8828735, 273.6366882, -380.4206543, 573.6169434, -733.8803101, 641.1923218
1: -181.5148926, 242.5971832, -415.9165344, 524.6207275, -700.8786011, 646.8203735
2: -181.4661713, 246.9006042, -414.8100586, 533.8364258, -709.9705811, 652.2425537
3: -214.6846161, 279.4453430, -485.4026794, 605.4370728, -813.9174194, 756.1090088
4: -184.8080750, 283.3780212, -416.2733459, 611.5494995, -794.1005249, 691.3298340

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_B2_B1

### Relational analysis result of IS_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9310407, upper bound: 398.8269876
time: 1.15 seconds

## Relational analysis of IS_B2_A1_B2_B2

### Relational analysis result of IS_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9337329, upper bound: 398.8847062
time: 1.25 seconds

## BFS IS instance: IS_B2_A2_B1

### Backsubstitution after applying IS history:
0: -381.7537842, 576.2619629, -371.6011658, 557.5308228, -916.3967285, 925.6267700
1: -417.4182434, 526.7597046, -405.9487610, 510.1193542, -905.2587891, 911.0247803
2: -416.3083801, 535.9980469, -404.9323730, 519.0952759, -914.6543579, 920.7594604
3: -487.2251282, 607.8753662, -473.1334534, 588.8765259, -1055.6387939, 1061.3710938
4: -417.8207397, 614.0283813, -405.7301636, 594.6139526, -996.9985962, 1004.6436768

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8221159, upper bound: 398.8637537
time: 1.02 seconds

## Relational analysis of IS_B2_A2_B1_A2

### Relational analysis result of IS_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8829510, upper bound: 398.8639063
time: 0.91 seconds

## BFS IS instance: IS_B2_A2_B2

### Backsubstitution after applying IS history:
0: -382.1474609, 576.9548950, -380.8507996, 574.4285889, -934.8452759, 935.9540405
1: -417.8582153, 527.3723145, -416.4064026, 525.2488403, -921.7328491, 922.3524780
2: -416.7462769, 536.6198730, -415.2940674, 534.4339600, -931.5006714, 932.1107788
3: -487.7610474, 608.5759277, -485.9724121, 606.1524658, -1074.5676270, 1075.1625977
4: -418.2799072, 614.7428589, -416.7651062, 612.2853394, -1015.8733521, 1016.7861328

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_B2_A1

### Relational analysis result of IS_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8221641, upper bound: 398.8828633
time: 1.15 seconds

## Relational analysis of IS_B2_A2_B2_A2

### Relational analysis result of IS_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8830196, upper bound: 398.8831541
time: 1.47 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 5.26 seconds
IS_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 5.26
Output dim: 0, lower bound: -398.9326858, upper bound: 398.8834642
IS_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 5.26
Output dim: 0, lower bound: -398.9327872, upper bound: 398.9064325
IS_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 5.26
Output dim: 0, lower bound: -398.8943235, upper bound: 398.9304202
IS_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 5.26
Output dim: 0, lower bound: -398.8942847, upper bound: 398.8942847
IS_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 5.26
Output dim: 0, lower bound: -398.8269982, upper bound: 398.9101895
IS_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 5.26
Output dim: 0, lower bound: -398.8823946, upper bound: 398.9154449
IS_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 5.26
Output dim: 0, lower bound: -398.8293560, upper bound: 398.9187599
IS_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 5.26
Output dim: 0, lower bound: -398.8848215, upper bound: 398.9196317
IS_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 5.26
Output dim: 0, lower bound: -398.9310214, upper bound: 398.8293560
IS_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 5.26
Output dim: 0, lower bound: -398.9333194, upper bound: 398.8814962
IS_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 5.26
Output dim: 0, lower bound: -398.9310407, upper bound: 398.8269876
IS_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 5.26
Output dim: 0, lower bound: -398.9337329, upper bound: 398.8847062
IS_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 5.26
Output dim: 0, lower bound: -398.8221159, upper bound: 398.8637537
IS_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 5.26
Output dim: 0, lower bound: -398.8829510, upper bound: 398.8639063
IS_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 5.26
Output dim: 0, lower bound: -398.8221641, upper bound: 398.8828633
IS_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 5.26
Output dim: 0, lower bound: -398.8830196, upper bound: 398.8831541

## BFS IS instance: IS_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -163.0847473, 270.6788940, -158.7276611, 256.1733704, -419.2581177, 429.4065552
1: -179.5193939, 239.9363251, -174.2021332, 227.3370819, -406.8564758, 414.1383667
2: -179.4758911, 244.1714783, -174.0921478, 231.2686768, -410.7445374, 418.2636108
3: -212.2688141, 276.3743896, -204.3563385, 261.9820862, -474.2509155, 480.7307129
4: -182.7720490, 280.2462158, -176.1709290, 266.0678406, -448.8398438, 456.4171448

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_B1_B1_B1

### Relational analysis result of IS_B1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8895666, upper bound: 398.8791164
time: 1.38 seconds

## Relational analysis of IS_B1_A1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B1_B1_A1

### Relational analysis result of IS_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9058942, upper bound: 398.8830825
time: 0.96 seconds

## Relational analysis of IS_B1_A1_B1_B1_A2

### Relational analysis result of IS_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9058942, upper bound: 398.8834642
time: 1.03 seconds

## BFS IS instance: IS_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -163.4986877, 271.4302979, -167.9532776, 274.1008301, -437.5995178, 439.3835144
1: -179.9843292, 240.6073303, -184.5478210, 243.2079163, -423.1922607, 425.1551514
2: -179.9349518, 244.8607941, -184.4354706, 247.6372070, -427.5721436, 429.2962646
3: -212.8421326, 277.1510925, -217.2633972, 280.2965088, -493.1386108, 494.4144592
4: -183.2584534, 281.0318298, -187.0465240, 284.5088806, -467.7673035, 468.0783691

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_B1_B2_B1

### Relational analysis result of IS_B1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8897338, upper bound: 398.9018730
time: 1.06 seconds

## Relational analysis of IS_B1_A1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B1_B2_A1

### Relational analysis result of IS_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9060508, upper bound: 398.9060508
time: 1.02 seconds

## Relational analysis of IS_B1_A1_B1_B2_A2

### Relational analysis result of IS_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9060508, upper bound: 398.9064325
time: 1.16 seconds

## BFS IS instance: IS_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -164.8828735, 273.6366882, -158.7444458, 261.9582825, -426.8411560, 432.3811340
1: -181.5148926, 242.5971832, -174.7173615, 232.4681091, -413.9830017, 417.3144226
2: -181.4661713, 246.9006042, -174.5723724, 236.7829437, -418.2489929, 421.4729309
3: -214.6846161, 279.4453430, -206.5065002, 267.8417664, -482.5263672, 485.9518127
4: -184.8080750, 283.3780212, -177.6686249, 271.7230225, -456.5310974, 461.0466309

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_A1_B2_B1_B1

### Relational analysis result of IS_B1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8920791, upper bound: 398.9004231
time: 1.09 seconds

## Relational analysis of IS_B1_A1_B2_B1_B2

### Relational analysis result of IS_B1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8923210, upper bound: 398.9290428
time: 1.05 seconds

## BFS IS instance: IS_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -163.9465637, 271.9315796, -180.5553436, 290.8768311, -454.8233948, 452.4869385
1: -180.4746704, 241.1051178, -198.1239929, 258.7351990, -439.2097778, 439.2291260
2: -180.4207153, 245.4162140, -198.2770386, 263.6422119, -444.0629272, 443.6932373
3: -213.4435272, 277.7338257, -233.0232544, 298.2536316, -511.6970520, 510.7570801
4: -183.7133484, 281.6534119, -200.6026001, 303.0548401, -486.7681885, 482.2560120

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B1_A1_B2_B2_B1

### Relational analysis result of IS_B1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8921358, upper bound: 398.8701408
time: 0.97 seconds

## Relational analysis of IS_B1_A1_B2_B2_B2

### Relational analysis result of IS_B1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8922516, upper bound: 398.8922516
time: 1.43 seconds

## BFS IS instance: IS_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -365.8552856, 550.9074707, -147.1558380, 238.8045349, -592.8819580, 693.9716187
1: -400.1148987, 504.1885376, -161.8349304, 213.0812531, -603.3856201, 661.2072754
2: -398.9710083, 512.8250732, -161.6179962, 217.5398712, -608.7635498, 669.4821167
3: -467.1267090, 581.7904663, -191.0690765, 245.5193787, -705.8071289, 767.2758789
4: -400.4382324, 587.9110718, -164.4872589, 249.4701996, -643.1698608, 750.5380859

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_B1_A1_A1

### Relational analysis result of IS_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8269982, upper bound: 398.9083081
time: 1.04 seconds

## Relational analysis of IS_B1_A2_B1_A1_A2

### Relational analysis result of IS_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8246330, upper bound: 398.9086246
time: 0.99 seconds

## BFS IS instance: IS_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -380.0779419, 573.1376953, -147.8710938, 239.7438202, -607.7517090, 716.8621826
1: -415.5257263, 524.1536865, -162.6090240, 213.9384460, -619.4958496, 681.9613037
2: -414.4352112, 533.3820190, -162.3888702, 218.4668427, -624.9581909, 690.7804565
3: -484.9528198, 604.9008179, -191.9472809, 246.5103302, -724.3591309, 791.0734863
4: -415.8859558, 611.0115356, -165.2416687, 250.5057068, -659.3604736, 774.3755493

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 3

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A2_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8300393, upper bound: 398.9079348
time: 1.03 seconds

## Relational analysis of IS_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_B1_A2_A1

### Relational analysis result of IS_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8790343, upper bound: 398.9128015
time: 1.01 seconds

## Relational analysis of IS_B1_A2_B1_A2_A2

### Relational analysis result of IS_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8822281, upper bound: 398.9086246
time: 3.34 seconds

## BFS IS instance: IS_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -365.0679321, 549.8778687, -171.1613159, 278.6345825, -632.3431396, 716.5551758
1: -399.2740479, 503.1655273, -188.6379242, 249.4654846, -637.8515625, 686.4457397
2: -398.0990601, 511.7638550, -188.3596191, 254.6869659, -643.6171875, 694.8668213
3: -466.1428223, 580.5615234, -223.3035583, 287.1181641, -745.6047363, 797.6050415
4: -399.6105957, 586.6911621, -192.3140106, 291.6528015, -683.7718506, 776.3916626

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_B2_A1_A1

### Relational analysis result of IS_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8293560, upper bound: 398.9174028
time: 1.13 seconds

## Relational analysis of IS_B1_A2_B2_A1_A2

### Relational analysis result of IS_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8269876, upper bound: 398.9175798
time: 1.13 seconds

## BFS IS instance: IS_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -379.3765869, 572.2478638, -171.9664307, 279.6744690, -647.3506470, 739.7218018
1: -414.7821960, 523.2603149, -189.5056915, 250.4682312, -654.0859985, 707.4517212
2: -413.6555481, 532.4514160, -189.2267456, 255.7161255, -659.9249268, 716.4329834
3: -484.0895691, 603.8063354, -224.2818756, 288.2797241, -764.3549194, 821.6366577
4: -415.1567078, 609.9378662, -193.1625366, 292.8448486, -700.1683960, 800.4567261

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A2_B2_A2_B1

### Relational analysis result of IS_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8324786, upper bound: 398.9157430
time: 1.11 seconds

## Relational analysis of IS_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_B2_A2_A1

### Relational analysis result of IS_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8814962, upper bound: 398.9175989
time: 0.94 seconds

## Relational analysis of IS_B1_A2_B2_A2_A2

### Relational analysis result of IS_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8847062, upper bound: 398.9183854
time: 0.95 seconds

## BFS IS instance: IS_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -162.7590485, 270.4719543, -368.8265381, 553.2884521, -711.4284668, 626.2604980
1: -179.1897125, 239.7384644, -403.1591187, 505.8085632, -679.5256958, 630.7223511
2: -179.1420288, 243.9651489, -402.1555481, 514.4545288, -688.0776978, 635.9970093
3: -211.9415588, 276.1505737, -470.0841064, 583.7122192, -789.4075928, 737.2312622
4: -182.4950409, 279.9880066, -402.9767456, 590.2501221, -770.3010864, 674.4506226

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A1_B1_B1_A1

### Relational analysis result of IS_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9083081, upper bound: 398.8269982
time: 1.25 seconds

## Relational analysis of IS_B2_A1_B1_B1_A2

### Relational analysis result of IS_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9174028, upper bound: 398.8293560
time: 1.08 seconds

## BFS IS instance: IS_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -163.4986877, 271.4302979, -381.0571899, 572.2239990, -731.0260620, 639.0941162
1: -179.9843292, 240.6073303, -416.3877258, 522.8253174, -697.3245239, 644.5762329
2: -179.9349518, 244.8607941, -415.4403687, 532.0175781, -706.3870850, 649.9434814
3: -212.8421326, 277.1510925, -485.3492126, 603.4548950, -809.8132324, 753.2918701
4: -183.2584534, 281.0318298, -416.1853943, 609.9800415, -790.7495117, 688.4595947

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 3

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A1_B1_B2_A1

### Relational analysis result of IS_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9291779, upper bound: 398.8324786
time: 1.19 seconds

## Relational analysis of IS_B2_A1_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_B1_B2_A1

### Relational analysis result of IS_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9289218, upper bound: 398.8703766
time: 1.00 seconds

## Relational analysis of IS_B2_A1_B1_B2_A2

### Relational analysis result of IS_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8919485, upper bound: 398.8703529
time: 1.08 seconds

## BFS IS instance: IS_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -164.1492615, 272.6777344, -365.8627625, 550.8289795, -710.4533691, 626.0063477
1: -180.7262421, 241.7296600, -400.1338806, 504.1309509, -679.6133423, 630.3779907
2: -180.6790466, 246.0066223, -398.9763489, 512.7404175, -688.1404419, 635.7269897
3: -213.7901764, 278.4461975, -467.1413574, 581.7271118, -789.5374756, 737.1383057
4: -184.0498657, 282.3369446, -400.4456177, 587.8389282, -769.7116089, 674.7671509

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A1_B2_B1_A1

### Relational analysis result of IS_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9086246, upper bound: 398.8246330
time: 1.01 seconds

## Relational analysis of IS_B2_A1_B2_B1_A2

### Relational analysis result of IS_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9175798, upper bound: 398.8269876
time: 0.99 seconds

## BFS IS instance: IS_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -164.8828735, 273.6366882, -380.1071472, 573.0991821, -733.3631592, 640.8685303
1: -181.5148926, 242.5971832, -415.5699158, 524.1550293, -700.4119873, 646.4643555
2: -181.4661713, 246.9006042, -414.4653320, 533.3591919, -709.4924927, 651.8888550
3: -214.6846161, 279.4453430, -484.9905090, 604.9036255, -813.3826904, 755.6873779
4: -184.8080750, 283.3780212, -415.9210510, 611.0021973, -793.5524292, 690.9724731

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A1_B2_B2_A1

### Relational analysis result of IS_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9140944, upper bound: 398.8822281
time: 0.92 seconds

## Relational analysis of IS_B2_A1_B2_B2_A2

### Relational analysis result of IS_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9183854, upper bound: 398.8847062
time: 1.03 seconds

## BFS IS instance: IS_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -367.4130554, 553.7146606, -370.7818604, 556.3150635, -900.6486816, 902.0150146
1: -401.8789673, 506.5465698, -405.0583801, 508.9969177, -888.3453979, 889.6630859
2: -400.7151794, 515.1730957, -404.0432129, 517.9458618, -897.6725464, 898.8122559
3: -469.2388000, 584.5025635, -472.1009216, 587.5765381, -1036.1510010, 1036.9869385
4: -402.2343750, 590.6312866, -404.8476562, 593.3059692, -980.2353516, 980.1443481

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A2_B1_A1_A1

### Relational analysis result of IS_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8221159, upper bound: 398.8629224
time: 1.03 seconds

## Relational analysis of IS_B2_A2_B1_A1_A2

### Relational analysis result of IS_B2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8208746, upper bound: 398.8634955
time: 0.94 seconds

## BFS IS instance: IS_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -381.6351929, 576.0594482, -371.6011658, 557.5308228, -916.2592773, 925.4133911
1: -417.2889099, 526.5825195, -405.9487610, 510.1193542, -905.1110229, 910.8363037
2: -416.1761169, 535.8190308, -404.9323730, 519.0952759, -914.5059814, 920.5689087
3: -487.0700684, 607.6740112, -473.1334534, 588.8765259, -1055.4674072, 1061.1568604
4: -417.6864014, 613.8228760, -405.7301636, 594.6139526, -996.8529663, 1004.4284058

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A2_B1_A2_B1

### Relational analysis result of IS_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8819502, upper bound: 398.8619206
time: 1.04 seconds

## Relational analysis of IS_B2_A2_B1_A2_B2

### Relational analysis result of IS_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8829510, upper bound: 398.8636482
time: 1.18 seconds

## BFS IS instance: IS_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -367.8015137, 554.3994751, -380.0375366, 573.2197266, -919.1188354, 912.3057251
1: -402.3133240, 507.1506958, -415.5222778, 524.1324463, -904.8347168, 900.9488525
2: -401.1470337, 515.7866211, -414.4111633, 533.2909546, -914.5343628, 910.1301270
3: -469.7665710, 585.1963501, -484.9473267, 604.8594360, -1055.0897217, 1050.7425537
4: -402.6873169, 591.3381348, -415.8879089, 610.9849854, -999.1121216, 992.2586670

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A2_B2_A1_A1

### Relational analysis result of IS_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8221641, upper bound: 398.8828230
time: 0.84 seconds

## Relational analysis of IS_B2_A2_B2_A1_A2

### Relational analysis result of IS_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8209228, upper bound: 398.8828633
time: 1.05 seconds

## BFS IS instance: IS_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -382.0289001, 576.7522583, -380.8507996, 574.4285889, -934.7077637, 935.7404785
1: -417.7287598, 527.1953125, -416.4064026, 525.2488403, -921.5850220, 922.1641235
2: -416.6140137, 536.4409790, -415.2940674, 534.4339600, -931.3520508, 931.9201050
3: -487.6061401, 608.3746338, -485.9724121, 606.1524658, -1074.3964844, 1074.9484863
4: -418.1456299, 614.5377197, -416.7651062, 612.2853394, -1015.7276611, 1016.5712891

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A2_B2_A2_B1

### Relational analysis result of IS_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8821026, upper bound: 398.8804847
time: 1.02 seconds

## Relational analysis of IS_B2_A2_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8830196, upper bound: 398.8831541
time: 1.12 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 5.64 seconds
IS_B1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.64
Output dim: 0, lower bound: -398.9058942, upper bound: 398.8830825
IS_B1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.64
Output dim: 0, lower bound: -398.9058942, upper bound: 398.8834642
IS_B1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.64
Output dim: 0, lower bound: -398.9060508, upper bound: 398.9060508
IS_B1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.64
Output dim: 0, lower bound: -398.9060508, upper bound: 398.9064325
IS_B1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 5.64
Output dim: 0, lower bound: -398.8920791, upper bound: 398.9004231
IS_B1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 5.64
Output dim: 0, lower bound: -398.8923210, upper bound: 398.9290428
IS_B1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 5.64
Output dim: 0, lower bound: -398.8921358, upper bound: 398.8701408
IS_B1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 5.64
Output dim: 0, lower bound: -398.8922516, upper bound: 398.8922516
IS_B1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 5.64
Output dim: 0, lower bound: -398.8269982, upper bound: 398.9083081
IS_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 5.64
Output dim: 0, lower bound: -398.8246330, upper bound: 398.9086246
IS_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 5.64
Output dim: 0, lower bound: -398.8790343, upper bound: 398.9128015
IS_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 5.64
Output dim: 0, lower bound: -398.8822281, upper bound: 398.9086246
IS_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 5.64
Output dim: 0, lower bound: -398.8293560, upper bound: 398.9174028
IS_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 5.64
Output dim: 0, lower bound: -398.8269876, upper bound: 398.9175798
IS_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 5.64
Output dim: 0, lower bound: -398.8814962, upper bound: 398.9175989
IS_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 5.64
Output dim: 0, lower bound: -398.8847062, upper bound: 398.9183854
IS_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.64
Output dim: 0, lower bound: -398.9083081, upper bound: 398.8269982
IS_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.64
Output dim: 0, lower bound: -398.9174028, upper bound: 398.8293560
IS_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.64
Output dim: 0, lower bound: -398.9289218, upper bound: 398.8703766
IS_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.64
Output dim: 0, lower bound: -398.8919485, upper bound: 398.8703529
IS_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.64
Output dim: 0, lower bound: -398.9086246, upper bound: 398.8246330
IS_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.64
Output dim: 0, lower bound: -398.9175798, upper bound: 398.8269876
IS_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.64
Output dim: 0, lower bound: -398.9140944, upper bound: 398.8822281
IS_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.64
Output dim: 0, lower bound: -398.9183854, upper bound: 398.8847062
IS_B2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 5.64
Output dim: 0, lower bound: -398.8221159, upper bound: 398.8629224
IS_B2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 5.64
Output dim: 0, lower bound: -398.8208746, upper bound: 398.8634955
IS_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.64
Output dim: 0, lower bound: -398.8819502, upper bound: 398.8619206
IS_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.64
Output dim: 0, lower bound: -398.8829510, upper bound: 398.8636482
IS_B2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 5.64
Output dim: 0, lower bound: -398.8221641, upper bound: 398.8828230
IS_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 5.64
Output dim: 0, lower bound: -398.8209228, upper bound: 398.8828633
IS_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.64
Output dim: 0, lower bound: -398.8821026, upper bound: 398.8804847
IS_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.64
Output dim: 0, lower bound: -398.8830196, upper bound: 398.8831541

## BFS IS instance: IS_B1_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -169.0375977, 276.0550537, -158.7276611, 256.1733704, -425.2109680, 434.7827148
1: -185.7464447, 244.9503021, -174.2021332, 227.3370819, -413.0835266, 419.1524353
2: -185.6264038, 249.3768158, -174.0921478, 231.2686768, -416.8950500, 423.4689636
3: -218.6567535, 282.2673645, -204.3563385, 261.9820862, -480.6388245, 486.6237183
4: -188.2801971, 286.4395752, -176.1709290, 266.0678406, -454.3480225, 462.6105042

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_B1_A1_B1_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8914364, upper bound: 398.8825831
time: 0.90 seconds

## Relational analysis of IS_B1_A1_B1_B1_A1_B2

### Relational analysis result of IS_B1_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8923038, upper bound: 398.8752255
time: 1.14 seconds

## BFS IS instance: IS_B1_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -162.3579102, 268.9609985, -158.7276611, 256.1733704, -418.5312805, 427.6886292
1: -178.7196045, 238.4770966, -174.2021332, 227.3370819, -406.0567017, 412.6791992
2: -178.6509705, 242.7531281, -174.0921478, 231.2686768, -409.9196167, 416.8452759
3: -211.3268127, 274.6976929, -204.3563385, 261.9820862, -473.3088989, 479.0540161
4: -181.8796539, 278.6767273, -176.1709290, 266.0678406, -447.9475098, 454.8476562

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_B1_A1_B1_B1_A2_B1

### Relational analysis result of IS_B1_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8914364, upper bound: 398.8827604
time: 0.99 seconds

## Relational analysis of IS_B1_A1_B1_B1_A2_B2

### Relational analysis result of IS_B1_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8923038, upper bound: 398.8763022
time: 1.16 seconds

## BFS IS instance: IS_B1_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -169.4530640, 276.8140869, -167.9532776, 274.1008301, -443.5538940, 444.7673645
1: -186.2131653, 245.6297760, -184.5478210, 243.2079163, -429.4210510, 430.1776123
2: -186.0870819, 250.0730896, -184.4354706, 247.6372070, -433.7243042, 434.5085144
3: -219.2395325, 283.0472107, -217.2633972, 280.2965088, -499.5360413, 500.3106079
4: -188.7692108, 287.2324219, -187.0465240, 284.5088806, -473.2780762, 474.2789307

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_B1_A1_B1_B2_A1_A1

### Relational analysis result of IS_B1_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9058848, upper bound: 398.8927413
time: 1.04 seconds

## Relational analysis of IS_B1_A1_B1_B2_A1_A2

### Relational analysis result of IS_B1_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8924627, upper bound: 398.8935464
time: 1.06 seconds

## BFS IS instance: IS_B1_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -162.7719269, 269.7117004, -167.9532776, 274.1008301, -436.8727417, 437.6649780
1: -179.1849213, 239.1481323, -184.5478210, 243.2079163, -422.3928223, 423.6959534
2: -179.1102448, 243.4538879, -184.4354706, 247.6372070, -426.7474365, 427.8893127
3: -211.9000854, 275.4799500, -217.2633972, 280.2965088, -492.1965332, 492.7433167
4: -182.3677216, 279.4620361, -187.0465240, 284.5088806, -466.8765259, 466.5085144

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B1_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_B1_A1_B1_B2_A2_B1

### Relational analysis result of IS_B1_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8915768, upper bound: 398.9060622
time: 1.20 seconds

## Relational analysis of IS_B1_A1_B1_B2_A2_B2

### Relational analysis result of IS_B1_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8924627, upper bound: 398.8940774
time: 1.04 seconds

## BFS IS instance: IS_B1_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -164.4686127, 272.8861084, -147.7432098, 240.7355347, -405.2041626, 420.6292725
1: -181.0496063, 241.9266052, -162.3699646, 213.7429810, -394.7926025, 404.2965088
2: -181.0067902, 246.2115784, -162.2663574, 217.4007111, -398.4075012, 408.4779358
3: -214.1112213, 278.6638794, -191.1653748, 246.2875519, -460.3987732, 469.8292542
4: -184.3210449, 282.5926819, -164.8296509, 249.8734436, -434.1944885, 447.4223328

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_B2_B1_B1_A1

### Relational analysis result of IS_B1_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8920791, upper bound: 398.9003573
time: 1.08 seconds

## Relational analysis of IS_B1_A1_B2_B1_B1_A2

### Relational analysis result of IS_B1_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8920791, upper bound: 398.9004231
time: 1.54 seconds

## BFS IS instance: IS_B1_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -164.8828735, 273.6366882, -157.2129669, 259.1912842, -424.0741577, 430.8496094
1: -181.5148926, 242.5971832, -173.0153656, 230.0614014, -411.5762939, 415.6124878
2: -181.4661713, 246.9006042, -172.8905029, 234.3733826, -415.8395081, 419.7911072
3: -214.6846161, 279.4453430, -204.4746857, 265.0773621, -479.7619629, 483.9199829
4: -184.8080750, 283.3780212, -175.9239197, 268.9533691, -453.7614441, 459.3019104

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_B2_B1_B2_A1

### Relational analysis result of IS_B1_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8923210, upper bound: 398.9290428
time: 1.32 seconds

## Relational analysis of IS_B1_A1_B2_B1_B2_A2

### Relational analysis result of IS_B1_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8923210, upper bound: 398.9290428
time: 0.95 seconds

## BFS IS instance: IS_B1_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -163.5345917, 271.1835327, -169.2099304, 270.1009216, -433.6354980, 440.3934021
1: -180.0117493, 240.4367371, -185.4367981, 240.2750854, -420.2868347, 425.8735046
2: -179.9637451, 244.7289734, -185.5826416, 244.8248749, -424.7886353, 430.3115845
3: -212.8728638, 276.9547729, -217.4822845, 277.0235596, -489.8964233, 494.4369812
4: -183.2285919, 280.8708496, -187.4414215, 281.4515686, -464.6801453, 468.3122253

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B1_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B1_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_B2_B2_B1_A1

### Relational analysis result of IS_B1_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8920791, upper bound: 398.8701408
time: 0.96 seconds

## Relational analysis of IS_B1_A1_B2_B2_B1_A2

### Relational analysis result of IS_B1_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8920791, upper bound: 398.8701408
time: 1.09 seconds

## BFS IS instance: IS_B1_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -163.9465637, 271.9315796, -179.0543213, 288.0926208, -452.0391846, 450.9859009
1: -180.4746704, 241.1051178, -196.4493713, 256.2946167, -436.7691650, 437.5544739
2: -180.4207153, 245.4162140, -196.6098022, 261.2218018, -441.6425171, 442.0260010
3: -213.4435272, 277.7338257, -230.9941864, 295.4341125, -508.8775635, 508.7279968
4: -183.7133484, 281.6534119, -198.8464050, 300.2438965, -483.9572449, 480.4998169

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B1_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B1_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_B2_B2_B2_A1

### Relational analysis result of IS_B1_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8922516, upper bound: 398.8922516
time: 1.00 seconds

## Relational analysis of IS_B1_A1_B2_B2_B2_A2

### Relational analysis result of IS_B1_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8922516, upper bound: 398.8922516
time: 1.14 seconds

## BFS IS instance: IS_B1_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -367.1645508, 550.2426147, -145.8205261, 236.6621246, -591.2658691, 691.7553101
1: -401.2845154, 503.2521362, -160.3587799, 211.1204529, -601.7138062, 658.4698486
2: -400.2980347, 511.8903503, -160.1304169, 215.4501343, -607.0328369, 666.7175293
3: -467.8351746, 580.7873535, -189.2929993, 243.2627258, -703.4782104, 764.1031494
4: -401.0507202, 587.2882690, -162.9549713, 247.1886139, -640.8186646, 748.0491943

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B1_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A2_B1_A1_A1_B1

### Relational analysis result of IS_B1_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8218060, upper bound: 398.9064379
time: 1.00 seconds

## Relational analysis of IS_B1_A2_B1_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A2_B1_A1_A1_B1

### Relational analysis result of IS_B1_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8265917, upper bound: 398.9056058
time: 1.23 seconds

## Relational analysis of IS_B1_A2_B1_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B1_A1_A1_A1

### Relational analysis result of IS_B1_A2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8123633, upper bound: 398.9071302
time: 1.10 seconds

## Relational analysis of IS_B1_A2_B1_A1_A1_A2

### Relational analysis result of IS_B1_A2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8197503, upper bound: 398.9072046
time: 0.93 seconds

## BFS IS instance: IS_B1_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -364.1054688, 547.6140137, -147.1558380, 238.8045349, -590.9567871, 690.5372925
1: -398.1564636, 501.4378967, -161.8349304, 213.0812531, -601.2728271, 658.3364868
2: -397.0115662, 510.0398560, -161.6179962, 217.5398712, -606.6458740, 666.5759277
3: -464.7667542, 578.6544800, -191.0690765, 245.5193787, -703.2869873, 763.9978027
4: -398.4158936, 584.7147217, -164.4872589, 249.4701996, -641.0050659, 747.2432861

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B1_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A2_B1_A1_A2_B1

### Relational analysis result of IS_B1_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8242278, upper bound: 398.9060037
time: 1.32 seconds

## Relational analysis of IS_B1_A2_B1_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B1_A1_A2_A1

### Relational analysis result of IS_B1_A2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8115109, upper bound: 398.9074567
time: 0.92 seconds

## Relational analysis of IS_B1_A2_B1_A1_A2_A2

### Relational analysis result of IS_B1_A2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8185091, upper bound: 398.9074969
time: 1.08 seconds

## BFS IS instance: IS_B1_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -379.1454163, 568.8988037, -146.5446320, 237.6089783, -603.9085693, 711.1051636
1: -414.2562256, 520.0226440, -161.1423645, 211.9850616, -615.3869019, 676.0513306
2: -413.3214417, 529.2020874, -160.9100342, 216.3863831, -620.8060303, 684.8007202
3: -482.8290405, 600.2543335, -190.1816101, 244.2621918, -719.3552856, 784.2615356
4: -414.0271301, 606.7259521, -163.7183990, 248.2337646, -654.6383057, 768.2639160

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 3

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A2_B1_A2_A1_B1

### Relational analysis result of IS_B1_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8300393, upper bound: 398.9064882
time: 0.93 seconds

## Relational analysis of IS_B1_A2_B1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A2_B1_A2_A1_B1

### Relational analysis result of IS_B1_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8679772, upper bound: 398.9089028
time: 1.19 seconds

## Relational analysis of IS_B1_A2_B1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B1_A2_A1_A1

### Relational analysis result of IS_B1_A2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8595066, upper bound: 398.9115368
time: 1.18 seconds

## Relational analysis of IS_B1_A2_B1_A2_A1_A2

### Relational analysis result of IS_B1_A2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8782742, upper bound: 398.9116804
time: 1.08 seconds

## BFS IS instance: IS_B1_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -378.4542847, 569.9804688, -147.8710938, 239.7438202, -605.9347534, 713.5595703
1: -413.7033691, 521.5709839, -162.6090240, 213.9384460, -617.5026855, 679.2623291
2: -412.6142273, 530.7656860, -162.3888702, 218.4668427, -622.9636230, 688.0469971
3: -482.7438660, 601.9618530, -191.9472809, 246.5103302, -722.0311890, 787.9969482
4: -414.0004883, 608.0092773, -165.2416687, 250.5057068, -657.3328857, 771.2672729

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B1_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A2_B1_A2_A2_B1

### Relational analysis result of IS_B1_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8809242, upper bound: 398.9107776
time: 0.95 seconds

## Relational analysis of IS_B1_A2_B1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_B1_A2_A2_A1

### Relational analysis result of IS_B1_A2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8611925, upper bound: 398.9129685
time: 1.18 seconds

## Relational analysis of IS_B1_A2_B1_A2_A2_A2

### Relational analysis result of IS_B1_A2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8808044, upper bound: 398.9130162
time: 1.26 seconds

## BFS IS instance: IS_B1_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -366.2861023, 549.0045166, -169.6657867, 276.3180847, -630.4724731, 713.9874878
1: -400.3375244, 502.0839539, -186.9928131, 247.2652435, -635.8926392, 683.4208984
2: -399.3172302, 510.6788330, -186.6997681, 252.4351654, -641.6203003, 691.7734985
3: -466.7189331, 579.3856812, -221.3470917, 284.5903320, -742.9447021, 794.0877686
4: -400.1103210, 585.8844604, -190.6219482, 289.1197510, -681.1166382, 773.5656738

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A2_B2_A1_A1_B1

### Relational analysis result of IS_B1_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8243047, upper bound: 398.9150843
time: 1.24 seconds

## Relational analysis of IS_B1_A2_B2_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A2_B2_A1_A1_B1

### Relational analysis result of IS_B1_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8289968, upper bound: 398.9137626
time: 1.29 seconds

## Relational analysis of IS_B1_A2_B2_A1_A1_B2

### Relational analysis result of IS_B1_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8289736, upper bound: 398.8916381
time: 1.37 seconds

## BFS IS instance: IS_B1_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -363.3897400, 546.6593018, -171.1613159, 278.6345825, -630.4826660, 713.1886597
1: -397.3892212, 500.4931335, -188.6379242, 249.4654846, -635.8055420, 683.6463623
2: -396.2125854, 509.0606995, -188.3596191, 254.6869659, -641.5662842, 692.0359497
3: -463.8668823, 577.5027466, -223.3035583, 287.1181641, -743.1627808, 794.4063721
4: -397.6592407, 583.5884399, -192.3140106, 291.6528015, -681.6722412, 773.1840210

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B1_A2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A2_B2_A1_A2_B1

### Relational analysis result of IS_B1_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8266284, upper bound: 398.9139340
time: 0.93 seconds

## Relational analysis of IS_B1_A2_B2_A1_A2_B2

### Relational analysis result of IS_B1_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8266035, upper bound: 398.8928135
time: 0.97 seconds

## BFS IS instance: IS_B1_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -379.0148926, 568.4293823, -170.4777527, 277.3634338, -643.8267212, 734.2084961
1: -414.1003723, 519.6177979, -187.8682404, 248.2749634, -650.3018188, 701.8613892
2: -413.1305237, 528.7678833, -187.5745850, 253.4719238, -656.1235352, 710.7443237
3: -482.5733643, 599.7375488, -222.3352966, 285.7597046, -759.6935425, 815.2021484
4: -413.8247375, 606.2332153, -191.4777832, 290.3210144, -695.7321777, 794.7557983

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 3

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A2_B2_A2_A1_B1

### Relational analysis result of IS_B1_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8324786, upper bound: 398.9143818
time: 1.07 seconds

## Relational analysis of IS_B1_A2_B2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A2_B2_A2_A1_B1

### Relational analysis result of IS_B1_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8703766, upper bound: 398.9138185
time: 1.00 seconds

## Relational analysis of IS_B1_A2_B2_A2_A1_B2

### Relational analysis result of IS_B1_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8703529, upper bound: 398.8915161
time: 0.99 seconds

## BFS IS instance: IS_B1_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -377.8147278, 569.1958618, -171.9664307, 279.6744690, -645.5941772, 736.5258179
1: -413.0296936, 520.7651367, -189.5056915, 250.4682312, -652.1614380, 704.8414307
2: -411.9026184, 529.9221802, -189.2267456, 255.7161255, -657.9978027, 713.7875977
3: -481.9605103, 600.9678955, -224.2818756, 288.2797241, -762.1045532, 818.6621704
4: -413.3397522, 607.0375366, -193.1625366, 292.8448486, -698.2088623, 797.4514160

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B1_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A2_B2_A2_A2_B1

### Relational analysis result of IS_B1_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8833863, upper bound: 398.9147751
time: 1.10 seconds

## Relational analysis of IS_B1_A2_B2_A2_A2_B2

### Relational analysis result of IS_B1_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8832446, upper bound: 398.8931493
time: 1.22 seconds

## BFS IS instance: IS_B2_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -145.8205261, 236.6621246, -367.1645508, 550.2426147, -691.7553101, 591.2658691
1: -160.3587799, 211.1204529, -401.2845764, 503.2521057, -658.4697876, 601.7138672
2: -160.1304169, 215.4501343, -400.2980652, 511.8903198, -666.7175293, 607.0328979
3: -189.2929993, 243.2627258, -467.8351746, 580.7872925, -764.1030884, 703.4782104
4: -162.9549713, 247.1886139, -401.0507202, 587.2882690, -748.0491943, 640.8186035

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A1_B1_B1_A1_A1

### Relational analysis result of IS_B2_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9064379, upper bound: 398.8218060
time: 1.08 seconds

## Relational analysis of IS_B2_A1_B1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_B1_B1_A1_A1

### Relational analysis result of IS_B2_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9056058, upper bound: 398.8265917
time: 1.42 seconds

## Relational analysis of IS_B2_A1_B1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_B1_B1_A1_B1

### Relational analysis result of IS_B2_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9071302, upper bound: 398.8123633
time: 1.32 seconds

## Relational analysis of IS_B2_A1_B1_B1_A1_B2

### Relational analysis result of IS_B2_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9072046, upper bound: 398.8197503
time: 1.25 seconds

## BFS IS instance: IS_B2_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -165.0623169, 270.6752319, -366.3404236, 549.0560913, -709.5745239, 624.8823853
1: -182.2064514, 242.1973572, -400.3997498, 502.1400146, -678.7659912, 630.8635864
2: -181.8080750, 247.0472870, -399.3758240, 510.7320862, -687.0366821, 636.3192139
3: -216.1772919, 278.7604980, -466.7888489, 579.4525146, -788.9647827, 737.1432495
4: -186.0498199, 282.9137573, -400.1690369, 585.9480591, -769.1457520, 674.9562378

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A1_B1_B1_A2_A1

### Relational analysis result of IS_B2_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9150843, upper bound: 398.8243047
time: 1.06 seconds

## Relational analysis of IS_B2_A1_B1_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_B1_B1_A2_A1

### Relational analysis result of IS_B2_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9137626, upper bound: 398.8289968
time: 0.98 seconds

## Relational analysis of IS_B2_A1_B1_B1_A2_A2

### Relational analysis result of IS_B2_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8916381, upper bound: 398.8289736
time: 0.99 seconds

## BFS IS instance: IS_B2_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -159.5236816, 263.7518921, -380.8506165, 571.9885864, -726.9265137, 631.1713257
1: -175.5693970, 233.9843292, -416.1647034, 522.6019897, -692.7651978, 637.8112183
2: -175.4503784, 238.2474518, -415.2205811, 531.7923584, -701.7521973, 643.1771851
3: -207.5092468, 269.5746155, -485.1029053, 603.1938477, -804.2750244, 745.4166260
4: -178.5932617, 273.3709106, -415.9747925, 609.7157593, -785.8675537, 680.5644531

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 3

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A1_B1_B2_A1_A1

### Relational analysis result of IS_B2_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9250960, upper bound: 398.8221699
time: 1.40 seconds

## Relational analysis of IS_B2_A1_B1_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_B1_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9273168, upper bound: 398.8528616
time: 0.89 seconds

## Relational analysis of IS_B2_A1_B1_B2_A1_B2

### Relational analysis result of IS_B2_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9274062, upper bound: 398.8655121
time: 1.28 seconds

## BFS IS instance: IS_B2_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -180.9696045, 292.2776184, -380.7465820, 571.1192627, -746.9538574, 659.2039795
1: -198.5952301, 259.8529053, -416.0034790, 521.9625854, -714.6998291, 663.8680420
2: -198.7601471, 264.7181702, -415.0455933, 531.1532593, -724.0905762, 669.7094727
3: -233.6329498, 299.5181885, -484.7943115, 602.4888306, -829.3856201, 775.4500732
4: -201.1019592, 304.2770386, -415.7062683, 608.9892578, -807.2443237, 711.1983643

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A1_B1_B2_A2_A1

### Relational analysis result of IS_B2_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8886389, upper bound: 398.8219575
time: 0.91 seconds

## Relational analysis of IS_B2_A1_B1_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_B1_B2_A2_B1

### Relational analysis result of IS_B2_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8895551, upper bound: 398.8528143
time: 1.38 seconds

## Relational analysis of IS_B2_A1_B1_B2_A2_B2

### Relational analysis result of IS_B2_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8896189, upper bound: 398.8654648
time: 0.93 seconds

## BFS IS instance: IS_B2_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -147.1558380, 238.8045349, -364.1054688, 547.6140137, -690.5372925, 590.9567261
1: -161.8349304, 213.0812531, -398.1564636, 501.4378967, -658.3364868, 601.2728271
2: -161.6179962, 217.5398712, -397.0115662, 510.0398560, -666.5758667, 606.6459351
3: -191.0690765, 245.5193787, -464.7667542, 578.6544800, -763.9978027, 703.2869873
4: -164.4872589, 249.4701996, -398.4158936, 584.7147217, -747.2432861, 641.0050659

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_B2_B1_A1_A1

### Relational analysis result of IS_B2_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9060037, upper bound: 398.8242278
time: 1.47 seconds

## Relational analysis of IS_B2_A1_B2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_B2_B1_A1_B1

### Relational analysis result of IS_B2_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9074567, upper bound: 398.8115109
time: 1.50 seconds

## Relational analysis of IS_B2_A1_B2_B1_A1_B2

### Relational analysis result of IS_B2_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9074969, upper bound: 398.8185091
time: 2.00 seconds

## BFS IS instance: IS_B2_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -166.5902557, 273.0410767, -363.4515381, 546.7181396, -708.8823242, 624.9291382
1: -183.8883057, 244.4338531, -397.4594727, 500.5568848, -679.0942383, 630.8139038
2: -183.4964447, 249.3477783, -396.2791748, 509.1212769, -687.3695068, 636.3294067
3: -218.1829834, 281.3241272, -463.9462891, 577.5789795, -789.3604736, 737.3975220
4: -187.7780151, 285.5042419, -397.7260132, 583.6608276, -768.8445435, 675.5741577

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_B2_B1_A2_A1

### Relational analysis result of IS_B2_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9139340, upper bound: 398.8266284
time: 1.35 seconds

## Relational analysis of IS_B2_A1_B2_B1_A2_A2

### Relational analysis result of IS_B2_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8928135, upper bound: 398.8266035
time: 1.49 seconds

## BFS IS instance: IS_B2_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -147.8710938, 239.7438202, -378.4542847, 569.9804688, -713.5595703, 605.9347534
1: -162.6090240, 213.9384460, -413.7033691, 521.5709839, -679.2622681, 617.5026855
2: -162.3888702, 218.4668427, -412.6142273, 530.7656860, -688.0469971, 622.9636230
3: -191.9472809, 246.5103302, -482.7438660, 601.9618530, -787.9969482, 722.0312500
4: -165.2416687, 250.5057068, -414.0004883, 608.0092773, -771.2673340, 657.3328247

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_B2_B2_A1_A1

### Relational analysis result of IS_B2_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9107776, upper bound: 398.8809242
time: 1.66 seconds

## Relational analysis of IS_B2_A1_B2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_B2_B2_A1_B1

### Relational analysis result of IS_B2_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9129685, upper bound: 398.8611925
time: 1.41 seconds

## Relational analysis of IS_B2_A1_B2_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9130162, upper bound: 398.8808044
time: 1.57 seconds

## BFS IS instance: IS_B2_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -171.9664307, 279.6744690, -377.8147278, 569.1959229, -736.5259399, 645.5941772
1: -189.5056915, 250.4682312, -413.0296936, 520.7651367, -704.8414917, 652.1614990
2: -189.2267456, 255.7161255, -411.9026184, 529.9221802, -713.7875977, 657.9977417
3: -224.2818756, 288.2797241, -481.9605408, 600.9678955, -818.6621094, 762.1046143
4: -193.1625366, 292.8448486, -413.3397522, 607.0375977, -797.4514771, 698.2088623

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_B2_B2_A2_A1

### Relational analysis result of IS_B2_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9147751, upper bound: 398.8833863
time: 1.32 seconds

## Relational analysis of IS_B2_A1_B2_B2_A2_A2

### Relational analysis result of IS_B2_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8931493, upper bound: 398.8832446
time: 1.26 seconds

## BFS IS instance: IS_B2_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -368.7950134, 553.0922241, -369.0243835, 553.3447266, -898.2046509, 899.4338379
1: -403.1339417, 505.6832886, -403.1106567, 506.2495422, -885.9678345, 886.5551147
2: -402.1142578, 514.3108521, -402.0950623, 515.1662598, -895.2672729, 895.6533813
3: -470.0361938, 583.5818481, -469.7585449, 584.4150391, -1032.9813232, 1033.3818359
4: -402.9245300, 590.0902100, -402.8372192, 590.1765747, -977.1260376, 977.3218384

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A2_B1_A1_A1_A1

### Relational analysis result of IS_B2_A2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8171517, upper bound: 398.8596490
time: 1.13 seconds

## Relational analysis of IS_B2_A2_B1_A1_A1_A2

### Relational analysis result of IS_B2_A2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8218710, upper bound: 398.8626864
time: 1.50 seconds

## BFS IS instance: IS_B2_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -365.7357788, 550.5073853, -370.7818604, 556.3150635, -898.7906494, 898.6633301
1: -399.9970093, 503.8777161, -405.0583801, 508.9969177, -886.3001709, 886.8696289
2: -398.8308411, 512.4702759, -404.0432129, 517.9458618, -895.6250610, 895.9837646
3: -466.9650879, 581.4497681, -472.1009216, 587.5765381, -1033.7121582, 1033.7960205
4: -400.2863159, 587.5310059, -404.8476562, 593.3059692, -978.1396484, 976.9424438

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B2_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A2_B1_A1_A2_B1

### Relational analysis result of IS_B2_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8195885, upper bound: 398.8632395
time: 1.11 seconds

## Relational analysis of IS_B2_A2_B1_A1_A2_B2

### Relational analysis result of IS_B2_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8206297, upper bound: 398.8632890
time: 1.06 seconds

## BFS IS instance: IS_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -379.8885498, 573.1312256, -372.3606262, 555.2705688, -912.0229492, 922.2653809
1: -415.3541260, 523.8607788, -406.4905090, 507.9357605, -900.7175293, 907.6362305
2: -414.2398071, 533.0621948, -405.6135864, 516.9199219, -910.0720825, 917.3480835
3: -484.7501831, 604.5399780, -473.0194092, 586.4747314, -1050.4216309, 1057.1113281
4: -415.6954956, 610.7197266, -405.5843506, 592.6271973, -992.6332397, 1000.4794922

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8580210, upper bound: 398.8110960
time: 1.66 seconds

## Relational analysis of IS_B2_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8759769, upper bound: 398.8517604
time: 2.04 seconds

## Relational analysis of IS_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8759716, upper bound: 398.8530077
time: 1.19 seconds

## BFS IS instance: IS_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -381.6351929, 576.0594482, -370.1048279, 554.5312500, -913.1168213, 923.7308960
1: -417.2889099, 526.5825195, -404.2666321, 507.6807556, -902.5579834, 908.9910889
2: -416.1761169, 535.8190308, -403.2489319, 516.6269531, -911.9216309, 918.7179565
3: -487.0700684, 607.6740112, -471.0787964, 586.1052246, -1052.5603027, 1058.9583740
4: -417.6864014, 613.8228760, -403.9668884, 591.7826538, -993.9136353, 1002.5199585

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8815979, upper bound: 398.8626229
time: 0.90 seconds

## Relational analysis of IS_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8814909, upper bound: 398.8634047
time: 1.30 seconds

## BFS IS instance: IS_B2_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -369.1825562, 553.7864990, -378.2922363, 570.2954712, -916.7083740, 909.7478638
1: -403.5670471, 506.2905884, -413.5891724, 521.4149780, -902.4655762, 897.8585205
2: -402.5456543, 514.9274292, -412.4765930, 530.5383301, -912.1318970, 906.9869385
3: -470.5653992, 584.2786255, -482.6302490, 601.7305298, -1051.9415283, 1047.1888428
4: -403.3775024, 590.8005981, -413.8993530, 607.8870239, -996.0263672, 989.4713745

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A2_B2_A1_A1_A1

### Relational analysis result of IS_B2_A2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8172561, upper bound: 398.8783478
time: 1.24 seconds

## Relational analysis of IS_B2_A2_B2_A1_A1_A2

### Relational analysis result of IS_B2_A2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8219207, upper bound: 398.8813824
time: 1.83 seconds

## BFS IS instance: IS_B2_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -366.1282349, 551.2015991, -380.0375366, 573.2197266, -917.2651367, 908.9634399
1: -400.4364624, 504.4904175, -415.5222778, 524.1324463, -902.7955933, 898.1644287
2: -399.2675476, 513.0921021, -414.4111633, 533.2909546, -912.4919434, 907.3098145
3: -467.4992981, 582.1501465, -484.9473267, 604.8594360, -1052.6577148, 1047.5607910
4: -400.7444153, 588.2474976, -415.8879089, 610.9849854, -997.0224609, 989.0661011

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B2_A2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A2_B2_A1_A2_B1

### Relational analysis result of IS_B2_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8198976, upper bound: 398.8817439
time: 1.47 seconds

## Relational analysis of IS_B2_A2_B2_A1_A2_B2

### Relational analysis result of IS_B2_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8206794, upper bound: 398.8816552
time: 1.40 seconds

## BFS IS instance: IS_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -380.2817383, 573.8234863, -381.3400269, 571.8204346, -930.1424561, 932.3097534
1: -415.7935486, 524.4732056, -416.6521301, 522.6293335, -916.7090454, 918.6560059
2: -414.6772461, 533.6834717, -415.6847229, 531.7957764, -926.4050903, 928.3945923
3: -485.2857056, 605.2402344, -485.5004883, 603.2637939, -1068.8186035, 1070.5969238
4: -416.1545105, 611.4338989, -416.3235779, 609.8165894, -1011.0253906, 1012.3642578

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8580210, upper bound: 398.8310571
time: 1.68 seconds

## Relational analysis of IS_B2_A2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8735057, upper bound: 398.8643219
time: 1.38 seconds

## Relational analysis of IS_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8735551, upper bound: 398.8656580
time: 1.10 seconds

## BFS IS instance: IS_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -382.0289001, 576.7522583, -379.2843323, 571.3497925, -931.4810181, 933.9791260
1: -417.7287598, 527.1953125, -414.6495667, 522.7396240, -918.9572754, 920.2342529
2: -416.6140137, 536.4409790, -413.5350037, 531.8937378, -928.6918945, 929.9862671
3: -487.6061401, 608.3746338, -483.8411560, 603.2985840, -1071.4023438, 1072.6939697
4: -418.1456299, 614.5377197, -414.9409790, 609.3745117, -1012.7084961, 1014.6034546

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=482.57733154296875
rel_dist={0: [-398.9373570313671, 398.9373570313671]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8852357, upper bound: 398.9232315
time: 1.47 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8851378, upper bound: 398.8851378
time: 1.31 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 3.02 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 3.02
Output dim: 0, lower bound: -398.8852357, upper bound: 398.9232315
IS_B2, status: Status.UNKNOWN, split count: 1, time: 3.02
Output dim: 0, lower bound: -398.8851378, upper bound: 398.8851378

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -179.2202911, 303.3570251, -164.8828735, 273.6366882, -452.8569946, 468.2398987
1: -197.4927673, 268.0154114, -181.5148926, 242.5971832, -440.0898438, 449.5303040
2: -197.7517548, 272.0600586, -181.4661713, 246.9006042, -444.6523438, 453.5261230
3: -234.1109924, 308.6250000, -214.6846161, 279.4453430, -513.5563354, 523.3095093
4: -201.8509827, 312.5909424, -184.8080750, 283.3780212, -485.2290039, 497.3990173

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 47

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8849102, upper bound: 398.8849102
time: 1.69 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8849102, upper bound: 398.8851378
time: 1.01 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -178.1441345, 301.5676880, -382.0378418, 576.7455444, -750.2157593, 670.8328247
1: -196.3163605, 266.4815674, -417.7334290, 527.1847534, -718.1521606, 672.4734497
2: -196.5529327, 270.5289307, -416.6237793, 536.4317627, -727.6116943, 677.2955322
3: -232.7352142, 306.8548584, -487.6061096, 608.3573608, -834.8328857, 785.7623901
4: -200.6661987, 310.7998352, -418.1519470, 614.5274048, -812.9034424, 720.6711426

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8851378, upper bound: 398.8849102
time: 1.11 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8851378, upper bound: 398.8851378
time: 2.08 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 5.94 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 5.94
Output dim: 0, lower bound: -398.8849102, upper bound: 398.8849102
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 5.94
Output dim: 0, lower bound: -398.8849102, upper bound: 398.8851378
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 5.94
Output dim: 0, lower bound: -398.8851378, upper bound: 398.8849102
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 5.94
Output dim: 0, lower bound: -398.8851378, upper bound: 398.8851378

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -164.8828735, 273.6366882, -164.8828735, 273.6366882, -438.5195618, 438.5195618
1: -181.5148926, 242.5971832, -181.5148926, 242.5971832, -424.1119995, 424.1119995
2: -181.4661713, 246.9006042, -181.4661713, 246.9006042, -428.3666992, 428.3666992
3: -214.6846161, 279.4453430, -214.6846161, 279.4453430, -494.1299438, 494.1299438
4: -184.8080750, 283.3780212, -184.8080750, 283.3780212, -468.1860962, 468.1860962

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8660929, upper bound: 398.9212790
time: 1.31 seconds

## Relational analysis of IS_B1_A1_A2

### Relational analysis result of IS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8838389, upper bound: 398.9220602
time: 1.46 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -381.8911133, 576.5324707, -164.8828735, 273.6366882, -642.8604126, 736.9483643
1: -417.5708313, 526.9865112, -181.5148926, 242.5971832, -648.6494751, 703.3659058
2: -416.4637146, 536.2330322, -181.4661713, 246.9006042, -654.0733643, 712.4891357
3: -487.4162598, 608.1259155, -214.6846161, 279.4453430, -758.2451172, 816.7497559
4: -417.9928894, 614.2974243, -184.8080750, 283.3780212, -693.1943359, 796.9583740

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_A1

### Relational analysis result of IS_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8821824, upper bound: 398.9201771
time: 1.26 seconds

## Relational analysis of IS_B1_A2_A2

### Relational analysis result of IS_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8850862, upper bound: 398.9222976
time: 1.77 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -164.8828735, 273.6366882, -381.8911133, 576.5324707, -736.9484253, 642.8603516
1: -181.5148926, 242.5971832, -417.5708313, 526.9865112, -703.3659058, 648.6494751
2: -181.4661713, 246.9006042, -416.4637146, 536.2330933, -712.4892578, 654.0733643
3: -214.6846161, 279.4453430, -487.4162598, 608.1259155, -816.7497559, 758.2451172
4: -184.8080750, 283.3780212, -417.9928894, 614.2974243, -796.9583740, 693.1943359

Time for backsubstitution: 2.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8832159, upper bound: 398.8818256
time: 1.28 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8847348, upper bound: 398.8847348
time: 1.48 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -382.1474609, 576.9548950, -382.1474609, 576.9548950, -937.3775635, 937.3775635
1: -417.8582153, 527.3723145, -417.8582153, 527.3723145, -923.8921509, 923.8921509
2: -416.7462769, 536.6198730, -416.7462769, 536.6198730, -933.7025757, 933.7026367
3: -487.7610474, 608.5759277, -487.7610474, 608.5759277, -1077.0296631, 1077.0296631
4: -418.2799072, 614.7428589, -418.2799072, 614.7428589, -1018.3641968, 1018.3641357

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8833310, upper bound: 398.8662508
time: 1.18 seconds

## Relational analysis of IS_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8834600, upper bound: 398.8835686
time: 1.63 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 7.46 seconds
IS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 7.46
Output dim: 0, lower bound: -398.8660929, upper bound: 398.9212790
IS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 7.46
Output dim: 0, lower bound: -398.8838389, upper bound: 398.9220602
IS_B1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 7.46
Output dim: 0, lower bound: -398.8821824, upper bound: 398.9201771
IS_B1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 7.46
Output dim: 0, lower bound: -398.8850862, upper bound: 398.9222976
IS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 7.46
Output dim: 0, lower bound: -398.8832159, upper bound: 398.8818256
IS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 7.46
Output dim: 0, lower bound: -398.8847348, upper bound: 398.8847348
IS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 7.46
Output dim: 0, lower bound: -398.8833310, upper bound: 398.8662508
IS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 7.46
Output dim: 0, lower bound: -398.8834600, upper bound: 398.8835686

## BFS IS instance: IS_B1_A1_A1

### Backsubstitution after applying IS history:
0: -153.7317200, 252.2772827, -164.1227417, 272.2572632, -425.9889526, 416.3999634
1: -169.0001678, 223.6766815, -180.6612854, 241.3658752, -410.3660278, 404.3379517
2: -168.9764252, 227.4005737, -180.6231232, 245.6348267, -414.6112366, 408.0236816
3: -199.1547699, 257.6705627, -213.6321716, 278.0136719, -477.1684570, 471.3027344
4: -171.7021637, 261.3429871, -183.9155731, 281.9352722, -453.6374207, 445.2585449

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_A1_A1

### Relational analysis result of IS_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8820711, upper bound: 398.9229097
time: 1.09 seconds

## Relational analysis of IS_B1_A1_A1_A2

### Relational analysis result of IS_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9015726, upper bound: 398.9293562
time: 1.10 seconds

## BFS IS instance: IS_B1_A1_A2

### Backsubstitution after applying IS history:
0: -163.3380432, 270.8208618, -164.8828735, 273.6366882, -436.9747314, 435.7037354
1: -179.7960052, 240.1278992, -181.5148926, 242.5971832, -422.3931274, 421.6427917
2: -179.7642059, 244.4613647, -181.4661713, 246.9006042, -426.6647949, 425.9274902
3: -212.6234283, 276.6451721, -214.6846161, 279.4453430, -492.0687561, 491.3297729
4: -183.0366974, 280.5787659, -184.8080750, 283.3780212, -466.4147339, 465.3868408

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_A2_B1

### Relational analysis result of IS_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9230026, upper bound: 398.9040734
time: 1.12 seconds

## Relational analysis of IS_B1_A1_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9296841, upper bound: 398.9296841
time: 1.39 seconds

## BFS IS instance: IS_B1_A2_A1

### Backsubstitution after applying IS history:
0: -381.3809814, 572.6707764, -162.0143280, 269.0518494, -637.0222778, 729.9821777
1: -416.7374878, 523.2285767, -178.3429413, 238.4609528, -642.6736450, 696.0705566
2: -415.7857056, 532.4358521, -178.2920532, 242.6643677, -647.9622192, 705.1567383
3: -485.7437134, 603.9161377, -210.8652649, 274.6812744, -751.1356812, 808.2819214
4: -416.5319519, 610.4614868, -181.5939789, 278.5072021, -686.2099609, 789.5390015

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A2_A1_A1

### Relational analysis result of IS_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8290736, upper bound: 398.9126628
time: 0.92 seconds

## Relational analysis of IS_B1_A2_A1_A2

### Relational analysis result of IS_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8805543, upper bound: 398.9199153
time: 0.95 seconds

## BFS IS instance: IS_B1_A2_A2

### Backsubstitution after applying IS history:
0: -380.2846680, 573.4118652, -164.8828735, 273.6366882, -641.0645142, 733.6823730
1: -415.7683716, 524.4342041, -181.5148926, 242.5971832, -646.6796875, 700.6976929
2: -414.6620789, 533.6459961, -181.4661713, 246.9006042, -652.1016846, 709.7851562
3: -485.2302551, 605.2210693, -214.6846161, 279.4453430, -755.9421997, 813.7081299
4: -416.1280823, 611.3295898, -184.8080750, 283.3780212, -691.1901245, 793.8846436

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A2_A2_A1

### Relational analysis result of IS_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8267939, upper bound: 398.9130968
time: 1.36 seconds

## Relational analysis of IS_B1_A2_A2_A2

### Relational analysis result of IS_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8832046, upper bound: 398.9220264
time: 0.94 seconds

## BFS IS instance: IS_B2_A1_B1

### Backsubstitution after applying IS history:
0: -162.0143280, 269.0518494, -381.3809814, 572.6707764, -729.9821777, 637.0222778
1: -178.3429413, 238.4609528, -416.7374878, 523.2285156, -696.0704956, 642.6736450
2: -178.2920532, 242.6643677, -415.7857056, 532.4359131, -705.1567383, 647.9622192
3: -210.8652649, 274.6812744, -485.7437134, 603.9161377, -808.2818604, 751.1356812
4: -181.5939789, 278.5072021, -416.5319519, 610.4614868, -789.5390015, 686.2100830

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_B1_B1

### Relational analysis result of IS_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9126628, upper bound: 398.8290736
time: 1.68 seconds

## Relational analysis of IS_B2_A1_B1_B2

### Relational analysis result of IS_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9199153, upper bound: 398.8805543
time: 1.37 seconds

## BFS IS instance: IS_B2_A1_B2

### Backsubstitution after applying IS history:
0: -164.8828735, 273.6366882, -380.2846680, 573.4118652, -733.6823730, 641.0644531
1: -181.5148926, 242.5971832, -415.7683716, 524.4342041, -700.6976929, 646.6796875
2: -181.4661713, 246.9006042, -414.6620789, 533.6459961, -709.7851562, 652.1016846
3: -214.6846161, 279.4453430, -485.2302551, 605.2210693, -813.7081299, 755.9422607
4: -184.8080750, 283.3780212, -416.1280823, 611.3295898, -793.8846436, 691.1901245

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_B2_B1

### Relational analysis result of IS_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9130968, upper bound: 398.8267939
time: 1.27 seconds

## Relational analysis of IS_B2_A1_B2_B2

### Relational analysis result of IS_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9220264, upper bound: 398.8832046
time: 0.89 seconds

## BFS IS instance: IS_B2_A2_B1

### Backsubstitution after applying IS history:
0: -381.4232788, 575.6804810, -371.6011658, 557.5308228, -916.0596313, 924.9574585
1: -417.0488892, 526.2454834, -405.9487610, 510.1193542, -904.8838501, 910.4498291
2: -415.9407959, 535.4760132, -404.9323730, 519.0952759, -914.2807617, 920.1599121
3: -486.7752380, 607.2877808, -473.1334534, 588.8765259, -1055.1750488, 1060.7104492
4: -417.4354858, 613.4286499, -405.7301636, 594.6139526, -996.5895386, 1003.9974976

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8218394, upper bound: 398.8636386
time: 1.15 seconds

## Relational analysis of IS_B2_A2_B1_A2

### Relational analysis result of IS_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8812362, upper bound: 398.8637830
time: 1.17 seconds

## BFS IS instance: IS_B2_A2_B2

### Backsubstitution after applying IS history:
0: -382.1474609, 576.9548950, -380.8507996, 574.4285889, -934.8452759, 935.9540405
1: -417.8582153, 527.3723145, -416.4064026, 525.2488403, -921.7328491, 922.3524780
2: -416.7462769, 536.6198730, -415.2940674, 534.4339600, -931.5006714, 932.1107788
3: -487.7610474, 608.5759277, -485.9724121, 606.1524658, -1074.5676270, 1075.1625977
4: -418.2799072, 614.7428589, -416.7651062, 612.2853394, -1015.8733521, 1016.7861328

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 6

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_B2_A1

### Relational analysis result of IS_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8219133, upper bound: 398.8811417
time: 1.41 seconds

## Relational analysis of IS_B2_A2_B2_A2

### Relational analysis result of IS_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8813666, upper bound: 398.8814435
time: 1.26 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 6.12 seconds
IS_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 6.12
Output dim: 0, lower bound: -398.8820711, upper bound: 398.9229097
IS_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 6.12
Output dim: 0, lower bound: -398.9015726, upper bound: 398.9293562
IS_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.12
Output dim: 0, lower bound: -398.9230026, upper bound: 398.9040734
IS_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.12
Output dim: 0, lower bound: -398.9296841, upper bound: 398.9296841
IS_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 6.12
Output dim: 0, lower bound: -398.8290736, upper bound: 398.9126628
IS_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 6.12
Output dim: 0, lower bound: -398.8805543, upper bound: 398.9199153
IS_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 6.12
Output dim: 0, lower bound: -398.8267939, upper bound: 398.9130968
IS_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 6.12
Output dim: 0, lower bound: -398.8832046, upper bound: 398.9220264
IS_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 6.12
Output dim: 0, lower bound: -398.9126628, upper bound: 398.8290736
IS_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 6.12
Output dim: 0, lower bound: -398.9199153, upper bound: 398.8805543
IS_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 6.12
Output dim: 0, lower bound: -398.9130968, upper bound: 398.8267939
IS_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 6.12
Output dim: 0, lower bound: -398.9220264, upper bound: 398.8832046
IS_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 6.12
Output dim: 0, lower bound: -398.8218394, upper bound: 398.8636386
IS_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 6.12
Output dim: 0, lower bound: -398.8812362, upper bound: 398.8637830
IS_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 6.12
Output dim: 0, lower bound: -398.8219133, upper bound: 398.8811417
IS_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 6.12
Output dim: 0, lower bound: -398.8813666, upper bound: 398.8814435

## BFS IS instance: IS_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: -158.7276611, 256.1733704, -161.2556152, 267.6744995, -426.4021301, 417.4289856
1: -174.2021332, 227.3370819, -177.4909058, 237.2307281, -411.4328613, 404.8280029
2: -174.0921478, 231.2686768, -177.4505005, 241.4007721, -415.4928894, 408.7190857
3: -204.3563385, 261.9820862, -209.8146362, 273.2695007, -477.6258545, 471.7967224
4: -176.1709290, 266.0678406, -180.7023926, 277.0673218, -453.2382507, 446.7702332

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_A1_A1_B1

### Relational analysis result of IS_B1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8820226, upper bound: 398.9038991
time: 1.23 seconds

## Relational analysis of IS_B1_A1_A1_A1_B2

### Relational analysis result of IS_B1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8820226, upper bound: 398.9229097
time: 1.11 seconds

## BFS IS instance: IS_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -151.6383972, 248.3205566, -164.1227417, 272.2572632, -423.8955994, 412.4432983
1: -166.6858826, 220.2291870, -180.6612854, 241.3658752, -408.0517578, 400.8904724
2: -166.6465302, 223.9093781, -180.6231232, 245.6348267, -412.2813416, 404.5324402
3: -196.3782043, 253.6978760, -213.6321716, 278.0136719, -474.3918457, 467.3300476
4: -169.3101044, 257.3973999, -183.9155731, 281.9352722, -451.2453613, 441.3129883

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_A1_A2_B1

### Relational analysis result of IS_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8970011, upper bound: 398.9038991
time: 0.94 seconds

## Relational analysis of IS_B1_A1_A1_A2_B2

### Relational analysis result of IS_B1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8970011, upper bound: 398.9293562
time: 1.02 seconds

## BFS IS instance: IS_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -160.4637604, 266.2216492, -169.4530640, 276.8140869, -437.2778320, 435.6747131
1: -176.6182404, 235.9741669, -186.2131653, 245.6297760, -422.2480164, 422.1873169
2: -176.5844269, 240.1839447, -186.0870819, 250.0730896, -426.6574707, 426.2709351
3: -208.8048859, 271.8652649, -219.2395325, 283.0472107, -491.8521118, 491.1047974
4: -179.8151245, 275.6994019, -188.7692108, 287.2324219, -467.0475464, 464.4686279

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_A2_B1_A1

### Relational analysis result of IS_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9040734, upper bound: 398.9040734
time: 0.89 seconds

## Relational analysis of IS_B1_A1_A2_B1_A2

### Relational analysis result of IS_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9040734, upper bound: 398.9040734
time: 1.07 seconds

## BFS IS instance: IS_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -163.3380432, 270.8208618, -162.7719269, 269.7117004, -433.0497437, 433.5927734
1: -179.7960052, 240.1278992, -179.1849213, 239.1481323, -418.9440918, 419.3128052
2: -179.7642059, 244.4613647, -179.1102448, 243.4538879, -423.2180786, 423.5715637
3: -212.6234283, 276.6451721, -211.9000854, 275.4799500, -488.1033630, 488.5452576
4: -183.0366974, 280.5787659, -182.3677216, 279.4620361, -462.4986877, 462.9464111

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_A2_B2_A1

### Relational analysis result of IS_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9040734, upper bound: 398.9230026
time: 0.93 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2

### Relational analysis result of IS_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9040734, upper bound: 398.9296841
time: 1.39 seconds

## BFS IS instance: IS_B1_A2_A1_A1

### Backsubstitution after applying IS history:
0: -367.5443420, 551.7126465, -160.3829193, 266.9401855, -621.5646973, 707.4598389
1: -401.7850037, 504.3478088, -176.5884705, 236.5487366, -626.1358032, 675.4287109
2: -400.7885132, 512.9641724, -176.5457306, 240.6895599, -631.2905273, 683.9765625
3: -468.5563354, 582.0181274, -208.8772278, 272.4777222, -732.0358887, 784.6313477
4: -401.6719055, 588.5042725, -179.9120178, 276.2057495, -669.3687744, 765.9436646

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A2_A1_A1_B1

### Relational analysis result of IS_B1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8236034, upper bound: 398.9099664
time: 1.08 seconds

## Relational analysis of IS_B1_A2_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_A1_A1_A1

### Relational analysis result of IS_B1_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8145238, upper bound: 398.9114364
time: 1.10 seconds

## Relational analysis of IS_B1_A2_A1_A1_A2

### Relational analysis result of IS_B1_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8218699, upper bound: 398.9115320
time: 1.32 seconds

## BFS IS instance: IS_B1_A2_A1_A2

### Backsubstitution after applying IS history:
0: -379.0308838, 569.7707520, -162.0143280, 269.0518494, -634.8675537, 727.1524658
1: -414.2273865, 520.5626831, -178.3429413, 238.4609528, -640.3417358, 693.4624023
2: -413.2830811, 529.7025757, -178.2920532, 242.6643677, -645.6304321, 702.4880981
3: -482.9590759, 600.8310547, -210.8652649, 274.6812744, -748.4746094, 805.2577515
4: -414.1431580, 607.2634277, -181.5939789, 278.5072021, -683.9359131, 786.3983154

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 3

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_A1_A2_A1

### Relational analysis result of IS_B1_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8613958, upper bound: 398.9186250
time: 1.27 seconds

## Relational analysis of IS_B1_A2_A1_A2_A2

### Relational analysis result of IS_B1_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8797516, upper bound: 398.9187979
time: 1.25 seconds

## BFS IS instance: IS_B1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -365.4333496, 550.2699585, -163.2835083, 271.5473633, -624.4953613, 709.0081787
1: -399.6715698, 503.6050415, -179.7948761, 240.7056122, -628.8879395, 678.1359863
2: -398.5189209, 512.2036133, -179.7502747, 244.9518280, -634.2102661, 686.6589966
3: -466.6178284, 581.1188354, -212.7328186, 277.2677307, -735.4723511, 787.8677979
4: -399.9996033, 587.2187500, -183.1551361, 281.1081238, -673.1256714, 768.1841431

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_A2_A1_A1

### Relational analysis result of IS_B1_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8137082, upper bound: 398.9115787
time: 0.91 seconds

## Relational analysis of IS_B1_A2_A2_A1_A2

### Relational analysis result of IS_B1_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8205923, upper bound: 398.9120746
time: 0.92 seconds

## BFS IS instance: IS_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -379.5524292, 572.2951050, -164.8828735, 273.6366882, -640.3518677, 732.5839844
1: -414.9657898, 523.4169312, -181.5148926, 242.5971832, -645.8952637, 699.6972046
2: -413.8635254, 532.6043091, -181.4661713, 246.9006042, -651.3212280, 708.7631836
3: -484.2920532, 604.0510254, -214.6846161, 279.4453430, -755.0148926, 812.5561523
4: -415.3340454, 610.1211548, -184.8080750, 283.3780212, -690.4076538, 792.6956787

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B1_A2_A2_A2_A1

### Relational analysis result of IS_B1_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8633814, upper bound: 398.9199908
time: 1.14 seconds

## Relational analysis of IS_B1_A2_A2_A2_A2

### Relational analysis result of IS_B1_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8817945, upper bound: 398.9208298
time: 1.01 seconds

## BFS IS instance: IS_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -160.3829193, 266.9401855, -367.5443420, 551.7126465, -707.4598389, 621.5646973
1: -176.5884705, 236.5487366, -401.7850037, 504.3478088, -675.4287109, 626.1358032
2: -176.5457306, 240.6895599, -400.7885132, 512.9641724, -683.9765015, 631.2905273
3: -208.8772278, 272.4777222, -468.5563354, 582.0181274, -784.6313477, 732.0359497
4: -179.9120178, 276.2057495, -401.6719055, 588.5042725, -765.9436646, 669.3687134

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A1_B1_B1_A1

### Relational analysis result of IS_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9099664, upper bound: 398.8236034
time: 1.19 seconds

## Relational analysis of IS_B2_A1_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_B1_B1_B1

### Relational analysis result of IS_B2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9114364, upper bound: 398.8145238
time: 1.41 seconds

## Relational analysis of IS_B2_A1_B1_B1_B2

### Relational analysis result of IS_B2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9115320, upper bound: 398.8218699
time: 1.09 seconds

## BFS IS instance: IS_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -162.0143280, 269.0518494, -379.0309143, 569.7707520, -727.1524658, 634.8676147
1: -178.3429413, 238.4609528, -414.2274170, 520.5626831, -693.4624023, 640.3416748
2: -178.2920532, 242.6643677, -413.2830811, 529.7025757, -702.4880371, 645.6304932
3: -210.8652649, 274.6812744, -482.9590759, 600.8309937, -805.2576904, 748.4746704
4: -181.5939789, 278.5072021, -414.1431580, 607.2634277, -786.3982544, 683.9359131

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 3

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_B1_B2_B1

### Relational analysis result of IS_B2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9186250, upper bound: 398.8613958
time: 1.21 seconds

## Relational analysis of IS_B2_A1_B1_B2_B2

### Relational analysis result of IS_B2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9187979, upper bound: 398.8797516
time: 0.98 seconds

## BFS IS instance: IS_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -163.2835083, 271.5473633, -365.4333801, 550.2699585, -709.0081787, 624.4953613
1: -179.7948761, 240.7056122, -399.6715393, 503.6050415, -678.1359863, 628.8878784
2: -179.7502747, 244.9518280, -398.5189209, 512.2036133, -686.6589966, 634.2103271
3: -212.7328186, 277.2677307, -466.6177979, 581.1188354, -787.8677368, 735.4723511
4: -183.1551361, 281.1081238, -399.9995728, 587.2187500, -768.1841431, 673.1256104

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_B2_B1_B1

### Relational analysis result of IS_B2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9115787, upper bound: 398.8137082
time: 1.17 seconds

## Relational analysis of IS_B2_A1_B2_B1_B2

### Relational analysis result of IS_B2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9120746, upper bound: 398.8205923
time: 0.98 seconds

## BFS IS instance: IS_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -164.8828735, 273.6366882, -379.5523987, 572.2951050, -732.5839844, 640.3518677
1: -181.5148926, 242.5971832, -414.9657593, 523.4169312, -699.6972046, 645.8953247
2: -181.4661713, 246.9006042, -413.8635254, 532.6043091, -708.7632446, 651.3212280
3: -214.6846161, 279.4453430, -484.2920532, 604.0509644, -812.5560913, 755.0149536
4: -184.8080750, 283.3780212, -415.3340454, 610.1210938, -792.6956177, 690.4076538

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_B2_B2_B1

### Relational analysis result of IS_B2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9199908, upper bound: 398.8633814
time: 1.33 seconds

## Relational analysis of IS_B2_A1_B2_B2_B2

### Relational analysis result of IS_B2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9208298, upper bound: 398.8817945
time: 1.14 seconds

## BFS IS instance: IS_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -367.0860596, 553.1387939, -369.7841492, 554.8323364, -898.8132935, 900.2962036
1: -401.5133362, 506.0382385, -403.9740906, 507.6288757, -886.5863037, 887.9608154
2: -400.3516846, 514.6568604, -402.9610596, 516.5449829, -895.8821411, 897.0886841
3: -468.7945862, 583.9190063, -470.8430176, 585.9920654, -1034.0950928, 1035.0465088
4: -401.8534546, 590.0368042, -403.7727051, 591.7124634, -978.2483521, 978.3922729

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 6

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A2_B1_A1_A1

### Relational analysis result of IS_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8218377, upper bound: 398.8628438
time: 1.04 seconds

## Relational analysis of IS_B2_A2_B1_A1_A2

### Relational analysis result of IS_B2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8205975, upper bound: 398.8633850
time: 1.29 seconds

## BFS IS instance: IS_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -381.3045654, 575.4777222, -371.6011658, 557.5308228, -915.9219360, 924.7437744
1: -416.9193115, 526.0681152, -405.9487610, 510.1193542, -904.7357788, 910.2612305
2: -415.8084717, 535.2968750, -404.9323730, 519.0952759, -914.1319580, 919.9691772
3: -486.6201782, 607.0860596, -473.1334534, 588.8765259, -1055.0037842, 1060.4957275
4: -417.3009949, 613.2230835, -405.7301636, 594.6139526, -996.4437866, 1003.7822876

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_B1_A2_B1

### Relational analysis result of IS_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -398.8279541, upper bound: 398.8154039
time: 1.17 seconds

## Relational analysis of IS_B2_A2_B1_A2_B2

### Relational analysis result of IS_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8279541, upper bound: 398.8637831
time: 1.17 seconds

## BFS IS instance: IS_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -367.8015137, 554.3994751, -379.0444641, 571.7432251, -917.6257935, 911.2519531
1: -402.3133240, 507.1506958, -414.4425964, 522.7687988, -903.4517822, 899.8137817
2: -401.1470337, 515.7866211, -413.3328552, 531.8947754, -913.1196289, 908.9990234
3: -469.7665710, 585.1963501, -483.6952820, 603.2802734, -1053.4912109, 1049.4516602
4: -402.6873169, 591.3381348, -414.8166809, 609.3966064, -997.5310059, 991.1439819

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A2_B2_A1_A1

### Relational analysis result of IS_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8219117, upper bound: 398.8810276
time: 1.08 seconds

## Relational analysis of IS_B2_A2_B2_A1_A2

### Relational analysis result of IS_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8206704, upper bound: 398.8811417
time: 1.33 seconds

## BFS IS instance: IS_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -382.0289001, 576.7522583, -380.8507996, 574.4285889, -934.7077637, 935.7404785
1: -417.7287598, 527.1953125, -416.4064026, 525.2488403, -921.5850220, 922.1641235
2: -416.6140137, 536.4409790, -415.2940674, 534.4339600, -931.3520508, 931.9201050
3: -487.6061401, 608.3746338, -485.9724121, 606.1524658, -1074.3964844, 1074.9484863
4: -418.1456299, 614.5377197, -416.7651062, 612.2853394, -1015.7276611, 1016.5712891

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A2_B2_A2_B1

### Relational analysis result of IS_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -398.8279770, upper bound: 398.8218173
time: 1.09 seconds

## Relational analysis of IS_B2_A2_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8279770, upper bound: 398.8814435
time: 1.05 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 5.83 seconds
IS_B1_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.83
Output dim: 0, lower bound: -398.8820226, upper bound: 398.9038991
IS_B1_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.83
Output dim: 0, lower bound: -398.8820226, upper bound: 398.9229097
IS_B1_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.83
Output dim: 0, lower bound: -398.8970011, upper bound: 398.9038991
IS_B1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.83
Output dim: 0, lower bound: -398.8970011, upper bound: 398.9293562
IS_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.83
Output dim: 0, lower bound: -398.9040734, upper bound: 398.9040734
IS_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.83
Output dim: 0, lower bound: -398.9040734, upper bound: 398.9040734
IS_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.83
Output dim: 0, lower bound: -398.9040734, upper bound: 398.9230026
IS_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.83
Output dim: 0, lower bound: -398.9040734, upper bound: 398.9296841
IS_B1_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 5.83
Output dim: 0, lower bound: -398.8145238, upper bound: 398.9114364
IS_B1_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 5.83
Output dim: 0, lower bound: -398.8218699, upper bound: 398.9115320
IS_B1_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 5.83
Output dim: 0, lower bound: -398.8613958, upper bound: 398.9186250
IS_B1_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 5.83
Output dim: 0, lower bound: -398.8797516, upper bound: 398.9187979
IS_B1_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 5.83
Output dim: 0, lower bound: -398.8137082, upper bound: 398.9115787
IS_B1_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 5.83
Output dim: 0, lower bound: -398.8205923, upper bound: 398.9120746
IS_B1_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 5.83
Output dim: 0, lower bound: -398.8633814, upper bound: 398.9199908
IS_B1_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 5.83
Output dim: 0, lower bound: -398.8817945, upper bound: 398.9208298
IS_B2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 5.83
Output dim: 0, lower bound: -398.9114364, upper bound: 398.8145238
IS_B2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 5.83
Output dim: 0, lower bound: -398.9115320, upper bound: 398.8218699
IS_B2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 5.83
Output dim: 0, lower bound: -398.9186250, upper bound: 398.8613958
IS_B2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 5.83
Output dim: 0, lower bound: -398.9187979, upper bound: 398.8797516
IS_B2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 5.83
Output dim: 0, lower bound: -398.9115787, upper bound: 398.8137082
IS_B2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 5.83
Output dim: 0, lower bound: -398.9120746, upper bound: 398.8205923
IS_B2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 5.83
Output dim: 0, lower bound: -398.9199908, upper bound: 398.8633814
IS_B2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 5.83
Output dim: 0, lower bound: -398.9208298, upper bound: 398.8817945
IS_B2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 5.83
Output dim: 0, lower bound: -398.8218377, upper bound: 398.8628438
IS_B2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 5.83
Output dim: 0, lower bound: -398.8205975, upper bound: 398.8633850
IS_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 5, time: 5.83
Output dim: 0, lower bound: -398.8279541, upper bound: 398.8154039
IS_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.83
Output dim: 0, lower bound: -398.8279541, upper bound: 398.8637831
IS_B2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 5.83
Output dim: 0, lower bound: -398.8219117, upper bound: 398.8810276
IS_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 5.83
Output dim: 0, lower bound: -398.8206704, upper bound: 398.8811417
IS_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 5, time: 5.83
Output dim: 0, lower bound: -398.8279770, upper bound: 398.8218173
IS_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.83
Output dim: 0, lower bound: -398.8279770, upper bound: 398.8814435

## BFS IS instance: IS_B1_A1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -158.7276611, 256.1733704, -168.6895142, 275.4176025, -434.1452637, 424.8628845
1: -174.2021332, 227.3370819, -185.3556824, 244.3800812, -418.5821533, 412.6927490
2: -174.0921478, 231.2686768, -185.2401581, 248.7922516, -422.8843994, 416.5087585
3: -204.3563385, 261.9820862, -218.1691589, 281.6129150, -485.9692383, 480.1512451
4: -176.1709290, 266.0678406, -187.8699341, 285.7739868, -461.9449158, 453.9377747

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_B1_A1_A1_A1_B1_A1

### Relational analysis result of IS_B1_A1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8805254, upper bound: 398.8906934
time: 1.22 seconds

## Relational analysis of IS_B1_A1_A1_A1_B1_A2

### Relational analysis result of IS_B1_A1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8750883, upper bound: 398.8915804
time: 0.99 seconds

## BFS IS instance: IS_B1_A1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -158.7276611, 256.1733704, -162.0113831, 268.3319702, -427.0596313, 418.1847229
1: -174.2021332, 227.3370819, -178.3302765, 237.9154510, -412.1175537, 405.6673584
2: -174.0921478, 231.2686768, -178.2662659, 242.1703644, -416.2624817, 409.5349426
3: -204.3563385, 261.9820862, -210.8472900, 274.0431519, -478.3994751, 472.8293152
4: -176.1709290, 266.0678406, -181.4716339, 278.0192566, -454.1901550, 447.5394592

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B1_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_B1_A1_A1_A1_B2_A1

### Relational analysis result of IS_B1_A1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8805254, upper bound: 398.8906934
time: 0.90 seconds

## Relational analysis of IS_B1_A1_A1_A1_B2_A2

### Relational analysis result of IS_B1_A1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8750883, upper bound: 398.8915804
time: 1.20 seconds

## BFS IS instance: IS_B1_A1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -151.6383972, 248.3205566, -168.6895142, 275.4176025, -427.0559692, 417.0100708
1: -166.6858826, 220.2291870, -185.3556824, 244.3800812, -411.0659485, 405.5848694
2: -166.6465302, 223.9093781, -185.2401581, 248.7922516, -415.4387817, 409.1494751
3: -196.3782043, 253.6978760, -218.1691589, 281.6129150, -477.9910583, 471.8670349
4: -169.3101044, 257.3973999, -187.8699341, 285.7739868, -455.0841064, 445.2673340

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_A1_A2_B1_B1

### Relational analysis result of IS_B1_A1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8660331, upper bound: 398.8955918
time: 1.09 seconds

## Relational analysis of IS_B1_A1_A1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_A1_A2_B1_A1

### Relational analysis result of IS_B1_A1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8872851, upper bound: 398.8815866
time: 1.04 seconds

## Relational analysis of IS_B1_A1_A1_A2_B1_A2

### Relational analysis result of IS_B1_A1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8970011, upper bound: 398.9038991
time: 1.32 seconds

## BFS IS instance: IS_B1_A1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -151.6383972, 248.3205566, -162.0113831, 268.3319702, -419.9703674, 410.3319397
1: -166.6858826, 220.2291870, -178.3302765, 237.9154510, -404.6013184, 398.5594482
2: -166.6465302, 223.9093781, -178.2662659, 242.1703644, -408.8168640, 402.1756592
3: -196.3782043, 253.6978760, -210.8472900, 274.0431519, -470.4213257, 464.5450745
4: -169.3101044, 257.3973999, -181.4716339, 278.0192566, -447.3293457, 438.8690186

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_A1_A2_B2_B1

### Relational analysis result of IS_B1_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8660331, upper bound: 398.9128580
time: 1.41 seconds

## Relational analysis of IS_B1_A1_A1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_B1_A1_A1_A2_B2_A1

### Relational analysis result of IS_B1_A1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8957599, upper bound: 398.8916249
time: 1.90 seconds

## Relational analysis of IS_B1_A1_A1_A2_B2_A2

### Relational analysis result of IS_B1_A1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8750697, upper bound: 398.8916791
time: 1.66 seconds

## BFS IS instance: IS_B1_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -167.9532776, 274.1008301, -169.4530640, 276.8140869, -444.7673340, 443.5538940
1: -184.5478210, 243.2079163, -186.2131653, 245.6297760, -430.1776123, 429.4210815
2: -184.4354706, 247.6372070, -186.0870819, 250.0730896, -434.5084839, 433.7243042
3: -217.2633972, 280.2965088, -219.2395325, 283.0472107, -500.3106079, 499.5360413
4: -187.0465240, 284.5088806, -188.7692108, 287.2324219, -474.2789307, 473.2780762

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_B1_A1_A2_B1_A1_B1

### Relational analysis result of IS_B1_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8908361, upper bound: 398.9023210
time: 1.04 seconds

## Relational analysis of IS_B1_A1_A2_B1_A1_B2

### Relational analysis result of IS_B1_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8917287, upper bound: 398.8923526
time: 0.95 seconds

## BFS IS instance: IS_B1_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -161.2266846, 266.9071960, -169.4530640, 276.8140869, -438.0407715, 436.3602600
1: -177.4663391, 236.6997681, -186.2131653, 245.6297760, -423.0960999, 422.9129333
2: -177.4091339, 241.0200500, -186.0870819, 250.0730896, -427.4822388, 427.1071167
3: -209.8498840, 272.6787109, -219.2395325, 283.0472107, -492.8970947, 491.9182129
4: -180.5965576, 276.6629028, -188.7692108, 287.2324219, -467.8289795, 465.4321289

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_B1_A1_A2_B1_A2_B1

### Relational analysis result of IS_B1_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8908361, upper bound: 398.9023210
time: 1.34 seconds

## Relational analysis of IS_B1_A1_A2_B1_A2_B2

### Relational analysis result of IS_B1_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8917287, upper bound: 398.8923526
time: 1.10 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -167.9532776, 274.1008301, -162.7719269, 269.7117004, -437.6649475, 436.8727417
1: -184.5478210, 243.2079163, -179.1849213, 239.1481323, -423.6959534, 422.3928223
2: -184.4354706, 247.6372070, -179.1102448, 243.4538879, -427.8892822, 426.7474365
3: -217.2633972, 280.2965088, -211.9000854, 275.4799500, -492.7433167, 492.1965027
4: -187.0465240, 284.5088806, -182.3677216, 279.4620361, -466.5085144, 466.8765259

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 47

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B1_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=482.57733154296875
rel_dist={0: [-398.93352538884415, 398.93352538884415]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 1092.73 seconds
