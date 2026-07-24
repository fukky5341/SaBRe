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
execution time: IAR + LP analysis = 2.48 + 2.42 = 4.90 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -398.9390144, upper bound: 398.9390144


# Binary Search by BASE starts (time budget: 1195.10 seconds, max iter: 100)

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
rel_dist={0: [-398.9255252454213, 398.92552524542134]}

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
rel_dist={0: [-398.9249620121283, 398.9249620142207]}

## Binary search (step 14) starts
Candidate diff: 0.0000305


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000305, mid=0.0000305, abs_max=482.57733154296875
rel_dist={0: [-398.9249573327561, 398.9249573331433]}

## Binary search (step 15) starts
Candidate diff: 0.0000153


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000153, mid=0.0000153, abs_max=482.57733154296875
rel_dist={0: [-398.9249549926937, 398.92495499269376]}

## Binary search (step 16) starts
Candidate diff: 0.0000076


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000076, mid=0.0000076, abs_max=482.57733154296875
rel_dist={0: [-398.9249538490367, 398.92495382392326]}

## Binary search (step 17) starts
Candidate diff: 0.0000038


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000038, mid=0.0000038, abs_max=482.57733154296875
rel_dist={0: [-398.9249532635105, 398.924953264239]}

## Binary search (step 18) starts
Candidate diff: 0.0000019


## IAR start
Binary search (step 18): status=Status.UNKNOWN, low=0.0000000, high=0.0000019, mid=0.0000019, abs_max=482.57733154296875
rel_dist={0: [-398.92495298950195, 398.92495302375687]}

## Binary search (step 19) starts
Candidate diff: 0.0000010


## IAR start
Binary search (step 19): status=Status.UNKNOWN, low=0.0000000, high=0.0000010, mid=0.0000010, abs_max=482.57733154296875
rel_dist={0: [-398.9249528754319, 398.92495289837075]}

## Binary Search Result
Binary search time: 98.49 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 1096.60 seconds

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9377021, upper bound: 398.8875484
time: 1.28 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8876411, upper bound: 398.8876411
time: 1.06 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.55 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.55
Output dim: 0, lower bound: -398.9377021, upper bound: 398.8875484
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.55
Output dim: 0, lower bound: -398.8876411, upper bound: 398.8876411

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -164.8828735, 273.6366882, -179.2202911, 303.3570251, -468.2398987, 452.8569946
1: -181.5148926, 242.5971832, -197.4927673, 268.0154114, -449.5303040, 440.0898438
2: -181.4661713, 246.9006042, -197.7517548, 272.0600586, -453.5261230, 444.6523438
3: -214.6846161, 279.4453430, -234.1109924, 308.6250000, -523.3095093, 513.5563354
4: -184.8080750, 283.3780212, -201.8509827, 312.5909424, -497.3990173, 485.2290039

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8874032, upper bound: 398.8874032
time: 1.17 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8874032, upper bound: 398.8874032
time: 0.97 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -382.1474609, 576.9548950, -178.9141998, 302.9039917, -672.2724609, 751.1802979
1: -417.8582153, 527.3723145, -197.1576538, 267.6112671, -673.7185669, 719.1681519
2: -416.7462769, 536.6198730, -197.4189301, 271.6546631, -678.5336914, 728.6610718
3: -487.7610474, 608.5759277, -233.7204895, 308.1575012, -787.2104492, 836.0215454
4: -418.2799072, 614.7428589, -201.5159454, 312.1154175, -722.1043091, 813.9605713

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8874032, upper bound: 398.8876411
time: 0.96 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8874032, upper bound: 398.8876411
time: 1.00 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.54 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.54
Output dim: 0, lower bound: -398.8874032, upper bound: 398.8874032
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.54
Output dim: 0, lower bound: -398.8874032, upper bound: 398.8874032
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.54
Output dim: 0, lower bound: -398.8874032, upper bound: 398.8876411
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.54
Output dim: 0, lower bound: -398.8874032, upper bound: 398.8876411

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -164.8828735, 273.6366882, -164.8828735, 273.6366882, -438.5195618, 438.5195618
1: -181.5148926, 242.5971832, -181.5148926, 242.5971832, -424.1119995, 424.1119995
2: -181.4661713, 246.9006042, -181.4661713, 246.9006042, -428.3666992, 428.3666992
3: -214.6846161, 279.4453430, -214.6846161, 279.4453430, -494.1299438, 494.1299438
4: -184.8080750, 283.3780212, -184.8080750, 283.3780212, -468.1860962, 468.1860962

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9332261, upper bound: 398.8856558
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8959646, upper bound: 398.8856499
time: 1.02 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -164.8828735, 273.6366882, -382.0796814, 576.8610229, -737.2622070, 643.0418091
1: -181.5148926, 242.5971832, -417.7807922, 527.2816162, -703.6503296, 648.8524170
2: -181.4661713, 246.9006042, -416.6720581, 536.5310059, -712.7783203, 654.2753906
3: -214.6846161, 279.4453430, -487.6701965, 608.4691162, -817.0795898, 758.4937744
4: -184.8080750, 283.3780212, -418.2040710, 614.6394653, -797.2938843, 693.3995972

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9332261, upper bound: 398.8856558
time: 1.15 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8959646, upper bound: 398.8856499
time: 1.15 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -382.0796814, 576.8610229, -164.8828735, 273.6366882, -643.0418091, 737.2621460
1: -417.7807922, 527.2816162, -181.5148926, 242.5971832, -648.8523560, 703.6502686
2: -416.6720581, 536.5310059, -181.4661713, 246.9006042, -654.2754517, 712.7783203
3: -487.6701965, 608.4691162, -214.6846161, 279.4453430, -758.4937744, 817.0795898
4: -418.2040710, 614.6394653, -184.8080750, 283.3780212, -693.3995972, 797.2938843

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8850404, upper bound: 398.8850291
time: 1.09 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8855706, upper bound: 398.8858878
time: 1.04 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -382.1474609, 576.9548950, -382.1474609, 576.9548950, -937.3775635, 937.3775635
1: -417.8582153, 527.3723145, -417.8582153, 527.3723145, -923.8921509, 923.8921509
2: -416.7462769, 536.6198730, -416.7462769, 536.6198730, -933.7025757, 933.7026367
3: -487.7610474, 608.5759277, -487.7610474, 608.5759277, -1077.0296631, 1077.0296631
4: -418.2799072, 614.7428589, -418.2799072, 614.7428589, -1018.3641968, 1018.3641357

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8850404, upper bound: 398.8850291
time: 1.16 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8855706, upper bound: 398.8858878
time: 1.10 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.83 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.83
Output dim: 0, lower bound: -398.9332261, upper bound: 398.8856558
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.83
Output dim: 0, lower bound: -398.8959646, upper bound: 398.8856499
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.83
Output dim: 0, lower bound: -398.9332261, upper bound: 398.8856558
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.83
Output dim: 0, lower bound: -398.8959646, upper bound: 398.8856499
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.83
Output dim: 0, lower bound: -398.8850404, upper bound: 398.8850291
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.83
Output dim: 0, lower bound: -398.8855706, upper bound: 398.8858878
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.83
Output dim: 0, lower bound: -398.8850404, upper bound: 398.8850291
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.83
Output dim: 0, lower bound: -398.8855706, upper bound: 398.8858878

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -160.8784637, 265.9215088, -164.8828735, 273.6366882, -434.5151062, 430.8043823
1: -177.0686951, 235.9393463, -181.5148926, 242.5971832, -419.6657104, 417.4541626
2: -176.9501801, 240.2561340, -181.4661713, 246.9006042, -423.8507690, 421.7222900
3: -209.3154907, 271.8285217, -214.6846161, 279.4453430, -488.7608337, 486.5131226
4: -180.1105499, 275.6749573, -184.8080750, 283.3780212, -463.4885864, 460.4830322

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8960425, upper bound: 398.8960425
time: 1.37 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8960425, upper bound: 398.8960425
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -182.5338745, 294.6768799, -164.5656738, 273.0683289, -455.6021729, 459.2425537
1: -200.3133240, 262.0391235, -181.1635132, 242.0994110, -442.4127197, 443.2026367
2: -200.4873199, 266.9307861, -181.1129150, 246.4041748, -446.8914795, 448.0436401
3: -235.6646729, 302.0430603, -214.2672882, 278.8741760, -514.5388184, 516.3103027
4: -202.8659515, 306.8235168, -184.4399414, 282.8017578, -485.6676025, 491.2634583

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8960425, upper bound: 398.8960439
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8960425, upper bound: 398.8960439
time: 1.32 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -160.8784637, 265.9215088, -382.0205688, 576.7855225, -733.2772217, 635.2211914
1: -177.0686951, 235.9393463, -417.7132263, 527.2068481, -699.1947632, 642.1936646
2: -176.9501801, 240.2561340, -416.6075745, 536.4582520, -708.2509766, 647.6141968
3: -209.3154907, 271.8285217, -487.5916443, 608.3809204, -811.6672974, 750.7367554
4: -180.1105499, 275.6749573, -418.1383057, 614.5540771, -792.5466309, 685.6030273

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8954231, upper bound: 398.8851182
time: 1.09 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8954231, upper bound: 398.8856485
time: 1.35 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -182.5338745, 294.6768799, -381.6713562, 576.1445312, -753.8096924, 663.3863525
1: -200.3133240, 262.0391235, -417.3336182, 526.6569214, -721.4659424, 668.3845825
2: -200.4873199, 266.9307861, -416.2171021, 535.9002075, -730.9154663, 674.2506714
3: -235.6646729, 302.0430603, -487.1379700, 607.7554321, -837.1154785, 781.0126343
4: -202.8659515, 306.8235168, -417.7417297, 613.9161987, -814.2782593, 716.4440918

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8954231, upper bound: 398.8851196
time: 1.24 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8954231, upper bound: 398.8856499
time: 0.95 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -379.3796387, 571.4879761, -164.8828735, 273.6366882, -640.0520630, 731.5964355
1: -414.7571411, 522.9238892, -181.5148926, 242.5971832, -645.5626831, 699.0456543
2: -413.6267700, 532.1519165, -181.4661713, 246.9006042, -650.9714355, 708.1559448
3: -483.9957581, 603.5317383, -214.6846161, 279.4453430, -754.6015625, 811.8581543
4: -415.0453491, 609.5784302, -184.8080750, 283.3780212, -690.0330811, 792.0319214

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8851182, upper bound: 398.8954231
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8851182, upper bound: 398.8954231
time: 1.03 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -404.0978088, 606.6862183, -164.5656738, 273.0683289, -661.7153320, 764.4280396
1: -441.5216980, 555.6407471, -181.1635132, 242.0994110, -669.5535278, 729.3895874
2: -440.4672546, 565.5830078, -181.1129150, 246.4041748, -675.1020508, 739.2628174
3: -514.7141724, 641.4462891, -214.2672882, 278.8741760, -782.9841919, 846.9870605
4: -441.1608887, 647.6649780, -184.4399414, 282.8017578, -713.8596802, 828.4184570

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8856485, upper bound: 398.8962818
time: 0.94 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8856485, upper bound: 398.8962818
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -379.4693604, 571.5990601, -382.1474609, 576.9548950, -934.4069214, 931.7293091
1: -414.8598328, 523.0347900, -417.8582153, 527.3723145, -920.6241455, 919.3082886
2: -413.7246399, 532.2597656, -416.7462769, 536.6198730, -930.4193115, 929.0993652
3: -484.1145325, 603.6630249, -487.7610474, 608.5759277, -1073.1617432, 1071.8328857
4: -415.1448059, 609.7048950, -418.2799072, 614.7428589, -1015.0192871, 1013.1253662

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8844989, upper bound: 398.8844989
time: 1.18 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8844989, upper bound: 398.8850291
time: 1.07 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -404.3253174, 607.0194092, -381.6713562, 576.1445312, -955.9311523, 964.5986938
1: -441.7807312, 555.9595947, -417.3336182, 526.6569214, -944.5192261, 949.6661377
2: -440.7174683, 565.8958130, -416.2171021, 535.9002075, -954.4462280, 960.2166748
3: -515.0227661, 641.8205566, -487.1379700, 607.7554321, -1101.4549561, 1106.9772949
4: -441.4168091, 648.0268555, -417.7417297, 613.9161987, -1038.7271729, 1049.5643311

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8850291, upper bound: 398.8853575
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8850291, upper bound: 398.8858877
time: 0.98 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.43 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 0, lower bound: -398.8960425, upper bound: 398.8960425
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 0, lower bound: -398.8960425, upper bound: 398.8960425
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 0, lower bound: -398.8960425, upper bound: 398.8960439
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 0, lower bound: -398.8960425, upper bound: 398.8960439
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 0, lower bound: -398.8954231, upper bound: 398.8851182
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 0, lower bound: -398.8954231, upper bound: 398.8856485
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 0, lower bound: -398.8954231, upper bound: 398.8851196
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 0, lower bound: -398.8954231, upper bound: 398.8856499
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 0, lower bound: -398.8851182, upper bound: 398.8954231
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 0, lower bound: -398.8851182, upper bound: 398.8954231
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 0, lower bound: -398.8856485, upper bound: 398.8962818
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 0, lower bound: -398.8856485, upper bound: 398.8962818
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 0, lower bound: -398.8844989, upper bound: 398.8844989
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 0, lower bound: -398.8844989, upper bound: 398.8850291
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 0, lower bound: -398.8850291, upper bound: 398.8853575
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 0, lower bound: -398.8850291, upper bound: 398.8858877

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -160.8784637, 265.9215088, -160.8784637, 265.9215088, -426.7999268, 426.7999268
1: -177.0686951, 235.9393463, -177.0686951, 235.9393463, -413.0078430, 413.0078430
2: -176.9501801, 240.2561340, -176.9501801, 240.2561340, -417.2062988, 417.2062988
3: -209.3154907, 271.8285217, -209.3154907, 271.8285217, -481.1440125, 481.1440125
4: -180.1105499, 275.6749573, -180.1105499, 275.6749573, -455.7854919, 455.7854919

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9153979, upper bound: 398.8936171
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9182663, upper bound: 398.8960247
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -160.8784637, 265.9215088, -182.5338745, 294.6768799, -455.5553284, 448.4553528
1: -177.0686951, 235.9393463, -200.3133240, 262.0391235, -439.1076965, 436.2526245
2: -176.9501801, 240.2561340, -200.4873199, 266.9307861, -443.8809814, 440.7434692
3: -209.3154907, 271.8285217, -235.6646729, 302.0430603, -511.3585510, 507.4931946
4: -180.1105499, 275.6749573, -202.8659515, 306.8235168, -486.9340515, 478.5408325

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9153979, upper bound: 398.8936171
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9182663, upper bound: 398.8960247
time: 1.12 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -182.5338745, 294.6768799, -160.8635864, 265.9017334, -448.4355469, 455.5404663
1: -200.3133240, 262.0391235, -177.0530396, 235.9219971, -436.2352905, 439.0921631
2: -200.4873199, 266.9307861, -176.9342804, 240.2380371, -440.7253418, 443.8650513
3: -235.6646729, 302.0430603, -209.2984161, 271.8085022, -507.4731750, 511.3414917
4: -202.8659515, 306.8235168, -180.0955811, 275.6540222, -478.5199585, 486.9190369

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8716315, upper bound: 398.8944705
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8945661, upper bound: 398.8945661
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -182.5338745, 294.6768799, -182.5338745, 294.6768799, -477.2107544, 477.2107544
1: -200.3133240, 262.0391235, -200.3133240, 262.0391235, -462.3524475, 462.3524475
2: -200.4873199, 266.9307861, -200.4873199, 266.9307861, -467.4180908, 467.4180908
3: -235.6646729, 302.0430603, -235.6646729, 302.0430603, -537.7077026, 537.7077026
4: -202.8659515, 306.8235168, -202.8659515, 306.8235168, -509.6894226, 509.6894226

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8716315, upper bound: 398.8944705
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8945661, upper bound: 398.8945661
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -160.8784637, 265.9215088, -379.3227539, 571.4169922, -727.6157227, 632.2337036
1: -177.0686951, 235.9393463, -414.6921387, 522.8531494, -694.5941162, 638.9066772
2: -176.9501801, 240.2561340, -413.5648193, 532.0831909, -703.6325684, 644.3128052
3: -209.3154907, 271.8285217, -483.9204712, 603.4481201, -806.4502563, 746.8475342
4: -180.1105499, 275.6749573, -414.9822693, 609.4979248, -787.2893677, 682.2396240

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9147785, upper bound: 398.8827180
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9176470, upper bound: 398.8851256
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -160.8784637, 265.9215088, -403.9565735, 606.5336304, -760.6790771, 654.3883057
1: -177.0686951, 235.9393463, -441.3610229, 555.4815063, -725.1992188, 663.3062744
2: -176.9501801, 240.2561340, -440.3142395, 565.4306641, -735.0058594, 668.8530273
3: -209.3154907, 271.8285217, -514.5303345, 641.2575073, -841.8902588, 775.7080078
4: -180.1105499, 275.6749573, -441.0070801, 647.4848633, -823.9447632, 706.5570068

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9147785, upper bound: 398.8832483
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9176470, upper bound: 398.8856558
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -182.5338745, 294.6768799, -379.4693604, 571.5990601, -748.9957886, 660.9020996
1: -200.3133240, 262.0391235, -414.8598328, 523.0347900, -717.6184082, 665.6520996
2: -200.4873199, 266.9307861, -413.7246399, 532.2597656, -727.0529175, 671.5055542
3: -235.6646729, 302.0430603, -484.1145325, 603.6630249, -832.7637329, 777.7787476
4: -202.8659515, 306.8235168, -415.1448059, 609.7048950, -809.8833008, 713.6464233

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8710516, upper bound: 398.8843358
time: 1.25 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8939862, upper bound: 398.8844314
time: 1.07 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -182.5338745, 294.6768799, -404.3006287, 606.9714355, -782.3065186, 683.2367554
1: -200.3133240, 262.0391235, -441.7525940, 555.9165039, -748.4707642, 690.2561035
2: -200.4873199, 266.9307861, -440.6898804, 565.8526611, -758.6668701, 696.2459106
3: -235.6646729, 302.0430603, -514.9876099, 641.7704468, -868.4945068, 806.8825073
4: -202.8659515, 306.8235168, -441.3879700, 647.9778442, -846.8217773, 738.1699219

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8710516, upper bound: 398.8843437
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8939862, upper bound: 398.8844378
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -379.3227539, 571.4169922, -160.8784637, 265.9215088, -632.2336426, 727.6157227
1: -414.6921082, 522.8531494, -177.0686951, 235.9393463, -638.9066772, 694.5941162
2: -413.5647888, 532.0832520, -176.9501801, 240.2561340, -644.3128052, 703.6326294
3: -483.9204712, 603.4481201, -209.3154907, 271.8285217, -746.8475952, 806.4503784
4: -414.9822693, 609.4978638, -180.1105499, 275.6749573, -682.2396240, 787.2893677

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8829690, upper bound: 398.8933374
time: 1.07 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8657263, upper bound: 398.8936693
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8844321, upper bound: 398.8939862
time: 1.26 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -379.4693604, 571.5990601, -182.5338745, 294.6768799, -660.9020996, 748.9958496
1: -414.8598328, 523.0347900, -200.3133240, 262.0391235, -665.6520386, 717.6184692
2: -413.7246399, 532.2597656, -200.4873199, 266.9307861, -671.5055542, 727.0529175
3: -484.1145325, 603.6630249, -235.6646729, 302.0430603, -777.7787476, 832.7637329
4: -415.1448059, 609.7048950, -202.8659515, 306.8235168, -713.6464233, 809.8832397

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8829690, upper bound: 398.8933374
time: 1.13 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8657263, upper bound: 398.8936693
time: 1.22 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8844321, upper bound: 398.8939862
time: 1.10 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -404.0398865, 606.6129150, -160.8635864, 265.9017334, -654.4414673, 760.7437744
1: -441.4557495, 555.5676880, -177.0530396, 235.9219971, -663.3717651, 725.2696533
2: -440.4040222, 565.5122070, -176.9342804, 240.2380371, -668.9153442, 735.0717163
3: -514.6371460, 641.3601074, -209.2984161, 271.8085022, -775.7829590, 841.9756470
4: -441.0968323, 647.5821533, -180.0955811, 275.6540222, -706.6191406, 824.0270386

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8838044, upper bound: 398.8941682
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8709600, upper bound: 398.8919706
time: 1.20 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -404.3006287, 606.9714355, -182.5338745, 294.6768799, -683.2367554, 782.3065186
1: -441.7525940, 555.9165039, -200.3133240, 262.0391235, -690.2560425, 748.4707642
2: -440.6898804, 565.8526611, -200.4873199, 266.9307861, -696.2459106, 758.6668701
3: -514.9876099, 641.7704468, -235.6646729, 302.0430603, -806.8825073, 868.4945068
4: -441.3879700, 647.9778442, -202.8659515, 306.8235168, -738.1699219, 846.8217773

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8838044, upper bound: 398.8941682
time: 0.91 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8709600, upper bound: 398.8919706
time: 1.19 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -379.4693604, 571.5990601, -379.4693604, 571.5990601, -928.7586060, 928.7586060
1: -414.8598328, 523.0347900, -414.8598328, 523.0347900, -916.0402222, 916.0402222
2: -413.7246399, 532.2597656, -413.7246399, 532.2597656, -925.8160400, 925.8160400
3: -484.1145325, 603.6630249, -484.1145325, 603.6630249, -1067.9649658, 1067.9648438
4: -415.1448059, 609.7048950, -415.1448059, 609.7048950, -1009.7804565, 1009.7804565

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8651464, upper bound: 398.8835346
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8838522, upper bound: 398.8838515
time: 1.33 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -379.4693604, 571.5990601, -404.3253174, 607.0194092, -962.1144409, 951.1172485
1: -414.8598328, 523.0347900, -441.7807312, 555.9595947, -946.9337158, 940.6716919
2: -413.7246399, 532.2597656, -440.7174683, 565.8958130, -957.4715576, 950.5836792
3: -484.1145325, 603.6630249, -515.0227661, 641.8205566, -1103.7432861, 1097.1033936
4: -415.1448059, 609.7048950, -441.4168091, 648.0268555, -1046.7667236, 1034.3321533

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8651464, upper bound: 398.8835425
time: 3.05 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8838522, upper bound: 398.8838594
time: 1.22 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -404.3253174, 607.0194092, -379.4693604, 571.5990601, -951.1172485, 962.1144409
1: -441.7807312, 555.9595947, -414.8598328, 523.0347900, -940.6716919, 946.9337158
2: -440.7174683, 565.8958130, -413.7246399, 532.2597656, -950.5836792, 957.4715576
3: -515.0227661, 641.8205566, -484.1145325, 603.6630249, -1097.1033936, 1103.7432861
4: -441.4168091, 648.0268555, -415.1448059, 609.7048950, -1034.3321533, 1046.7667236

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8653481, upper bound: 398.8846040
time: 1.18 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8838593, upper bound: 398.8846500
time: 1.08 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -404.3253174, 607.0194092, -404.3253174, 607.0194092, -984.4731445, 984.4731445
1: -441.7807312, 555.9595947, -441.7807312, 555.9595947, -971.5652466, 971.5652466
2: -440.7174683, 565.8958130, -440.7174683, 565.8958130, -982.2391968, 982.2391968
3: -515.0227661, 641.8205566, -515.0227661, 641.8205566, -1132.8818359, 1132.8818359
4: -441.4168091, 648.0268555, -441.4168091, 648.0268555, -1071.3186035, 1071.3186035

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8653481, upper bound: 398.8846119
time: 1.11 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8838593, upper bound: 398.8846579
time: 1.10 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 5.36 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.9153979, upper bound: 398.8936171
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.9182663, upper bound: 398.8960247
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.9153979, upper bound: 398.8936171
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.9182663, upper bound: 398.8960247
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8716315, upper bound: 398.8944705
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8945661, upper bound: 398.8945661
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8716315, upper bound: 398.8944705
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8945661, upper bound: 398.8945661
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.9147785, upper bound: 398.8827180
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.9176470, upper bound: 398.8851256
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.9147785, upper bound: 398.8832483
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.9176470, upper bound: 398.8856558
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8710516, upper bound: 398.8843358
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8939862, upper bound: 398.8844314
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8710516, upper bound: 398.8843437
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8939862, upper bound: 398.8844378
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8657263, upper bound: 398.8936693
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8844321, upper bound: 398.8939862
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8657263, upper bound: 398.8936693
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8844321, upper bound: 398.8939862
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8838044, upper bound: 398.8941682
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8709600, upper bound: 398.8919706
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8838044, upper bound: 398.8941682
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8709600, upper bound: 398.8919706
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8651464, upper bound: 398.8835346
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8838522, upper bound: 398.8838515
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8651464, upper bound: 398.8835425
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8838522, upper bound: 398.8838594
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8653481, upper bound: 398.8846040
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8838593, upper bound: 398.8846500
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8653481, upper bound: 398.8846119
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8838593, upper bound: 398.8846579

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -143.9629669, 232.2097778, -160.8784637, 265.9215088, -409.8844604, 393.0882263
1: -158.2770081, 207.5072632, -177.0686951, 235.9393463, -394.2162781, 384.5758362
2: -157.9999237, 212.2220764, -176.9501801, 240.2561340, -398.2560425, 389.1722412
3: -186.7086945, 239.1380768, -209.3154907, 271.8285217, -458.5372314, 448.4535522
4: -160.7478485, 243.0447235, -180.1105499, 275.6749573, -436.4227905, 423.1552734

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9129977, upper bound: 398.9129977
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9129977, upper bound: 398.9158662
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -168.5615845, 272.8850708, -160.5561981, 265.4156189, -433.9771729, 433.4412842
1: -185.7120361, 244.9842834, -176.7199707, 235.5006104, -421.2126465, 421.7042236
2: -185.3814392, 250.2069092, -176.5976105, 239.8125916, -425.1940308, 426.8045044
3: -219.6568146, 281.9783020, -208.9131165, 271.3230286, -490.9798584, 490.8914185
4: -189.2194977, 286.4551392, -179.7661591, 275.1594543, -464.3789673, 466.2213135

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9158662, upper bound: 398.9154052
time: 0.99 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9158662, upper bound: 398.9182737
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -143.9629669, 232.2097778, -182.5338745, 294.6768799, -438.6398315, 414.7436523
1: -158.2770081, 207.5072632, -200.3133240, 262.0391235, -420.3161316, 407.8205872
2: -157.9999237, 212.2220764, -200.4873199, 266.9307861, -424.9306030, 412.7094116
3: -186.7086945, 239.1380768, -235.6646729, 302.0430603, -488.7517395, 474.8027344
4: -160.7478485, 243.0447235, -202.8659515, 306.8235168, -467.5713501, 445.9106750

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9152035, upper bound: 398.8691338
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9152991, upper bound: 398.8921741
time: 1.65 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -168.5615845, 272.8850708, -182.1030121, 294.0525513, -462.6140442, 454.9880676
1: -185.7120361, 244.9842834, -199.8495331, 261.4844666, -447.1965027, 444.8338013
2: -185.3814392, 250.2069092, -200.0200653, 266.3709717, -451.7523804, 450.2269897
3: -219.6568146, 281.9783020, -235.1327515, 301.3973389, -521.0541382, 517.1110840
4: -189.2194977, 286.4551392, -202.4039917, 306.1695862, -495.3890686, 488.8591309

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9181313, upper bound: 398.8715492
time: 1.27 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9182269, upper bound: 398.8945896
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -171.1241913, 273.7840576, -160.8635864, 265.9017334, -437.0259094, 434.6476440
1: -187.5600433, 243.4932709, -177.0530396, 235.9219971, -423.4820557, 420.5463257
2: -187.7182159, 247.8797760, -176.9342804, 240.2380371, -427.9562378, 424.8140564
3: -220.0472717, 280.7072754, -209.2984161, 271.8085022, -491.8557739, 490.0056763
4: -189.6266632, 285.1177368, -180.0955811, 275.6540222, -465.2807007, 465.2132568

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8691338, upper bound: 398.9152035
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8715492, upper bound: 398.9181313
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -181.0280762, 291.8794556, -160.8635864, 265.9017334, -446.9297791, 452.7430420
1: -198.6312103, 259.5796204, -177.0530396, 235.9219971, -434.5531311, 436.6326599
2: -198.8130798, 264.4493713, -176.9342804, 240.2380371, -439.0510864, 441.3836670
3: -233.6219635, 299.2090454, -209.2984161, 271.8085022, -505.4304810, 508.5074463
4: -201.1061249, 303.9932556, -180.0955811, 275.6540222, -476.7601318, 484.0888062

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8921741, upper bound: 398.9152991
time: 1.50 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8945896, upper bound: 398.9182269
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -171.1241913, 273.7840576, -182.5338745, 294.6768799, -465.8010864, 456.3179016
1: -187.5600433, 243.4932709, -200.3133240, 262.0391235, -449.5991821, 443.8065491
2: -187.7182159, 247.8797760, -200.4873199, 266.9307861, -454.6489868, 448.3670959
3: -220.0472717, 280.7072754, -235.6646729, 302.0430603, -522.0903320, 516.3719482
4: -189.6266632, 285.1177368, -202.8659515, 306.8235168, -496.4501343, 487.9836121

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8715256, upper bound: 398.8715359
time: 1.14 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8715256, upper bound: 398.8944705
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -181.0280762, 291.8794556, -182.5338745, 294.6768799, -475.7049561, 474.4133301
1: -198.6312103, 259.5796204, -200.3133240, 262.0391235, -460.6702271, 459.8929443
2: -198.8130798, 264.4493713, -200.4873199, 266.9307861, -465.7437744, 464.9367065
3: -233.6219635, 299.2090454, -235.6646729, 302.0430603, -535.6648560, 534.8737183
4: -201.1061249, 303.9932556, -202.8659515, 306.8235168, -507.9296265, 506.8591919

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8944602, upper bound: 398.8716315
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8944602, upper bound: 398.8945661
time: 2.97 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -143.9629669, 232.2097778, -379.0955200, 571.1829834, -710.9205933, 599.2509155
1: -158.2770081, 207.5072632, -414.4322815, 522.6057129, -676.0311890, 611.9815063
2: -157.9999237, 212.2220764, -413.3195190, 531.8464966, -684.7972412, 617.3940430
3: -186.7086945, 239.1380768, -483.6268005, 603.1535645, -783.9781494, 715.7050781
4: -160.7478485, 243.0447235, -414.7353516, 609.2165527, -768.0036011, 650.7579956

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9147759, upper bound: 398.8813378
time: 1.07 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9144023, upper bound: 398.8633344
time: 1.26 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9147192, upper bound: 398.8820402
time: 1.09 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -168.5615845, 272.8850708, -378.9829102, 570.8166504, -734.7001953, 639.9543457
1: -185.7120361, 244.9842834, -414.3163147, 522.3424072, -702.5989990, 647.8622437
2: -185.3814392, 250.2069092, -413.1836853, 531.5634766, -711.5457764, 653.6614990
3: -219.6568146, 281.9783020, -483.4710693, 602.8565674, -815.8493042, 757.2070312
4: -189.2194977, 286.4551392, -414.5987549, 608.8980713, -795.3267212, 692.9715576

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9168983, upper bound: 398.8835678
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9173301, upper bound: 398.8657498
time: 1.23 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9176470, upper bound: 398.8844556
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -143.9629669, 232.2097778, -403.7302856, 606.2981567, -743.9823608, 621.4065552
1: -158.2770081, 207.5072632, -441.1036072, 555.2327271, -706.6347046, 636.3828735
2: -157.9999237, 212.2220764, -440.0694885, 565.1934814, -716.1697388, 641.9350586
3: -186.7086945, 239.1380768, -514.2371216, 640.9621582, -819.4172363, 744.5662231
4: -160.7478485, 243.0447235, -440.7617188, 647.2039795, -804.6592407, 675.0767822

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9156067, upper bound: 398.8821731
time: 1.08 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9134090, upper bound: 398.8693288
time: 1.19 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -168.5615845, 272.8850708, -403.6263733, 605.9648438, -767.7952271, 662.1195679
1: -185.7120361, 244.9842834, -440.9973145, 554.9932251, -733.2265625, 672.2747192
2: -185.3814392, 250.2069092, -439.9449463, 564.9337769, -742.9415894, 678.2144165
3: -219.6568146, 281.9783020, -514.0972900, 640.6916504, -851.3150024, 786.0838013
4: -189.2194977, 286.4551392, -440.6379700, 646.9114990, -832.0088501, 717.3038940

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9177291, upper bound: 398.8844032
time: 1.17 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9134090, upper bound: 398.8715588
time: 1.35 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -171.1241913, 273.7840576, -379.3422241, 571.4172363, -737.5864258, 638.8941040
1: -187.5600433, 243.4932709, -414.7145081, 522.8602905, -704.6696777, 646.2990723
2: -187.7182159, 247.8797760, -413.5850525, 532.0884399, -714.2080688, 651.5242920
3: -220.0472717, 280.7072754, -483.9428406, 603.4573975, -816.8653564, 755.5133057
4: -189.6266632, 285.1177368, -415.0016785, 609.5054321, -796.2124023, 691.4205933

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8649391, upper bound: 398.8823471
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8707347, upper bound: 398.8656307
time: 1.09 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8707347, upper bound: 398.8843366
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -181.0280762, 291.8794556, -379.4693604, 571.5990601, -747.4999390, 658.1854858
1: -198.6312103, 259.5796204, -414.8598328, 523.0347900, -715.9424438, 663.2575073
2: -198.8130798, 264.4493713, -413.7246399, 532.2597656, -725.3631592, 669.1027222
3: -233.6219635, 299.2090454, -484.1145325, 603.6630249, -830.7346191, 775.0203857
4: -201.1061249, 303.9932556, -415.1448059, 609.7048950, -808.1529541, 710.9124756

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8911499, upper bound: 398.8825325
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8936693, upper bound: 398.8657263
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8936693, upper bound: 398.8844321
time: 1.12 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -171.1241913, 273.7840576, -403.9720154, 606.5288696, -770.6452026, 661.0446167
1: -187.5600433, 243.4932709, -441.3783875, 555.4833984, -735.2697144, 670.6943359
2: -187.7182159, 247.8797760, -440.3299866, 565.4306641, -745.5764160, 676.0601807
3: -220.0472717, 280.7072754, -514.5472412, 641.2606812, -852.2996826, 784.3688965
4: -189.6266632, 285.1177368, -441.0218811, 647.4866943, -832.8623047, 715.7340088

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8649391, upper bound: 398.8825553
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8328200, upper bound: 398.8663197
time: 1.33 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -181.0280762, 291.8794556, -404.2758179, 606.9310913, -780.7722168, 680.4965820
1: -198.6312103, 259.5796204, -441.7243042, 555.8788452, -746.7584839, 687.8347778
2: -198.8130798, 264.4493713, -440.6623840, 565.8154907, -756.9409790, 693.8167725
3: -233.6219635, 299.2090454, -514.9533081, 641.7264404, -866.4230347, 804.0913086
4: -201.1061249, 303.9932556, -441.3596497, 647.9349365, -845.0492554, 735.4088745

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8911499, upper bound: 398.8827268
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8726940, upper bound: 398.8664053
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -368.9217224, 552.1719360, -160.8784637, 265.9215088, -621.4450073, 707.2002563
1: -402.9424438, 505.7899170, -177.0686951, 235.9393463, -626.8962402, 676.5793457
2: -401.9102173, 514.7539062, -176.9501801, 240.2561340, -632.2587891, 685.2066040
3: -469.4589844, 583.9561157, -209.3154907, 271.8285217, -732.1480713, 785.8010254
4: -402.5606384, 589.5466309, -180.1105499, 275.6749573, -669.4146118, 766.5913086

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8633344, upper bound: 398.9144023
time: 1.14 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8657498, upper bound: 398.9173301
time: 0.98 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -377.9890137, 568.8236694, -160.8784637, 265.9215088, -630.7728882, 725.0172729
1: -413.1980591, 520.6695557, -177.0686951, 235.9393463, -637.3255005, 692.3754883
2: -412.0708313, 529.8396606, -176.9501801, 240.2561340, -642.6787109, 701.3737793
3: -482.0799561, 600.9557495, -209.3154907, 271.8285217, -744.9282837, 803.9197998
4: -413.4230042, 606.9719238, -180.1105499, 275.6749573, -680.6169434, 784.7301636

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8820402, upper bound: 398.9147192
time: 1.07 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8844556, upper bound: 398.9176470
time: 1.19 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -369.0663452, 552.3657227, -182.5338745, 294.6768799, -650.1131592, 728.5898438
1: -403.1070862, 505.9794922, -200.3133240, 262.0391235, -653.6409912, 699.6099243
2: -402.0683594, 514.9389648, -200.4873199, 266.9307861, -659.4517822, 708.6343384
3: -469.6511230, 584.1793823, -235.6646729, 302.0430603, -763.0805054, 812.1209106
4: -402.7211609, 589.7617798, -202.8659515, 306.8235168, -700.8206787, 789.1927490

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8656307, upper bound: 398.8707347
time: 1.08 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8656307, upper bound: 398.8936693
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -378.1610107, 569.0414429, -182.5338745, 294.6768799, -659.4648438, 746.4318237
1: -413.3947754, 520.8857422, -200.3133240, 262.0391235, -664.0975952, 715.4334106
2: -412.2584534, 530.0501709, -200.4873199, 266.9307861, -669.8975220, 724.8274536
3: -482.3082886, 601.2113037, -235.6646729, 302.0430603, -775.8914185, 830.2728882
4: -413.6139221, 607.2186279, -202.8659515, 306.8235168, -712.0509033, 807.3633423

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8843366, upper bound: 398.8710516
time: 1.01 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8843366, upper bound: 398.8939862
time: 1.08 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -392.5766296, 587.8374023, -160.8635864, 265.9017334, -642.3000488, 741.4711914
1: -428.6576538, 538.4776611, -177.0530396, 235.9219971, -649.9899902, 707.7532959
2: -427.6835938, 548.3002319, -176.9342804, 240.2380371, -655.6302490, 717.3722534
3: -499.4528198, 621.8140259, -209.2984161, 271.8085022, -760.1712646, 821.9742432
4: -427.9679565, 627.6954346, -180.0955811, 275.6540222, -693.1561279, 803.7295532

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8821731, upper bound: 398.9156067
time: 0.90 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8835678, upper bound: 398.9177291
time: 1.08 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -401.1458130, 601.4669800, -160.8635864, 265.9017334, -651.4567871, 755.5133057
1: -438.2283936, 551.0467529, -177.0530396, 235.9219971, -660.0701294, 720.7041626
2: -437.1665955, 560.9525146, -176.9342804, 240.2380371, -665.6014404, 730.4818115
3: -510.7526245, 636.1943359, -209.2984161, 271.8085022, -771.8558350, 836.7315063
4: -437.7670593, 642.3484497, -180.0955811, 275.6540222, -703.2329102, 818.8201294

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8693288, upper bound: 398.9134090
time: 0.98 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8715588, upper bound: 398.9155315
time: 1.08 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -392.8669128, 588.2491455, -182.5338745, 294.6768799, -671.1237183, 763.0902710
1: -428.9877014, 538.8761597, -200.3133240, 262.0391235, -676.9059448, 731.0066528
2: -428.0027771, 548.6917725, -200.4873199, 266.9307861, -682.9935303, 741.0220337
3: -499.8457642, 622.2802124, -235.6646729, 302.0430603, -791.3125610, 848.5527344
4: -428.2945251, 628.1505737, -202.8659515, 306.8235168, -724.7418213, 826.5874023

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8826879, upper bound: 398.8657021
time: 1.01 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8828733, upper bound: 398.8919129
time: 1.06 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -401.4080505, 601.8262329, -182.5338745, 294.6768799, -680.2536621, 777.0770264
1: -438.5265808, 551.3973389, -200.3133240, 262.0391235, -686.9550171, 743.9072876
2: -437.4541321, 561.2950439, -200.4873199, 266.9307861, -692.9337158, 754.0791626
3: -511.1046448, 636.6063843, -235.6646729, 302.0430603, -802.9579468, 863.2520752
4: -438.0592041, 642.7460938, -202.8659515, 306.8235168, -734.7846680, 841.6169434

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8674129, upper bound: 398.8635723
time: 1.13 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8675982, upper bound: 398.8897921
time: 0.94 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -369.0663452, 552.3657227, -379.4693604, 571.5990601, -917.9696045, 908.3526611
1: -403.1070862, 505.9794922, -414.8598328, 523.0347900, -904.0291138, 898.0317383
2: -402.0683594, 514.9389648, -413.7246399, 532.2597656, -913.7623291, 907.3974609
3: -469.6511230, 584.1793823, -484.1145325, 603.6630249, -1053.2669678, 1047.3221436
4: -402.7211609, 589.7617798, -415.1448059, 609.7048950, -996.9548340, 989.0899048

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8648295, upper bound: 398.8648295
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8648295, upper bound: 398.8835354
time: 1.06 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -378.1610107, 569.0414429, -379.4693604, 571.5990601, -927.3214111, 926.1946411
1: -413.3947754, 520.8857422, -414.8598328, 523.0347900, -914.4857178, 913.8551636
2: -412.2584534, 530.0501709, -413.7246399, 532.2597656, -924.2080688, 923.5906372
3: -482.3082886, 601.2113037, -484.1145325, 603.6630249, -1066.0776367, 1065.4739990
4: -413.6139221, 607.2186279, -415.1448059, 609.7048950, -1008.1848755, 1007.2605591

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8835354, upper bound: 398.8651464
time: 0.91 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8835354, upper bound: 398.8838522
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -369.0663452, 552.3657227, -404.3253174, 607.0194092, -951.3253784, 930.7113037
1: -403.1070862, 505.9794922, -441.7807312, 555.9595947, -934.9226074, 922.6632080
2: -402.0683594, 514.9389648, -440.7174683, 565.8958130, -945.4178467, 932.1650391
3: -469.6511230, 584.1793823, -515.0227661, 641.8205566, -1089.0452881, 1076.4604492
4: -402.7211609, 589.7617798, -441.4168091, 648.0268555, -1033.9410400, 1013.6414795

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8655892, upper bound: 398.8650312
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8655892, upper bound: 398.8835425
time: 1.15 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -378.1610107, 569.0414429, -404.3253174, 607.0194092, -960.6771240, 948.5532837
1: -413.3947754, 520.8857422, -441.7807312, 555.9595947, -945.3793335, 938.4866333
2: -412.2584534, 530.0501709, -440.7174683, 565.8958130, -955.8635864, 948.3582764
3: -482.3082886, 601.2113037, -515.0227661, 641.8205566, -1101.8562012, 1094.6125488
4: -413.6139221, 607.2186279, -441.4168091, 648.0268555, -1045.1713867, 1031.8121338

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8842951, upper bound: 398.8653481
time: 1.18 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8842951, upper bound: 398.8838594
time: 1.05 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -393.9781494, 588.2157593, -379.4693604, 571.5990601, -940.3660889, 942.1171265
1: -430.0381470, 539.1370850, -414.8598328, 523.0347900, -928.6704102, 929.1774292
2: -429.1344604, 548.8555298, -413.7246399, 532.2597656, -938.5838623, 939.3298340
3: -500.5832214, 622.5844727, -484.1145325, 603.6630249, -1082.4013672, 1083.3776855
4: -428.9787292, 628.2936401, -415.1448059, 609.7048950, -1021.4697266, 1026.2910156

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8650312, upper bound: 398.8658989
time: 1.38 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8650312, upper bound: 398.8846048
time: 1.17 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -403.0519409, 604.5369263, -379.4693604, 571.5990601, -949.7177124, 959.6243896
1: -440.3587952, 553.8668213, -414.8598328, 523.0347900, -939.1625366, 944.8048096
2: -439.2883301, 563.7449951, -413.7246399, 532.2597656, -949.0139160, 955.3044434
3: -513.2688599, 639.4283447, -484.1145325, 603.6630249, -1095.2695312, 1101.3065186
4: -439.9255066, 645.6051025, -415.1448059, 609.7048950, -1032.7783203, 1044.3210449

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8677452, upper bound: 398.8813125
time: 0.93 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8835425, upper bound: 398.8659449
time: 1.09 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8835425, upper bound: 398.8846508
time: 1.25 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -393.9781494, 588.2157593, -404.3253174, 607.0194092, -973.7219849, 964.4758301
1: -430.0381470, 539.1370850, -441.7807312, 555.9595947, -959.5639038, 953.8089600
2: -429.1344604, 548.8555298, -440.7174683, 565.8958130, -970.2393799, 964.0974121
3: -500.5832214, 622.5844727, -515.0227661, 641.8205566, -1118.1799316, 1112.5162354
4: -428.9787292, 628.2936401, -441.4168091, 648.0268555, -1058.4562988, 1050.8426514

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8653824, upper bound: 398.8661006
time: 1.27 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8653824, upper bound: 398.8846119
time: 1.21 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -403.0519409, 604.5369263, -404.3253174, 607.0194092, -983.0735474, 981.9830322
1: -440.3587952, 553.8668213, -441.7807312, 555.9595947, -970.0560303, 969.4362793
2: -439.2883301, 563.7449951, -440.7174683, 565.8958130, -980.6693726, 980.0720825
3: -513.2688599, 639.4283447, -515.0227661, 641.8205566, -1131.0478516, 1130.4451904
4: -439.9255066, 645.6051025, -441.4168091, 648.0268555, -1069.7647705, 1068.8728027

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8682573, upper bound: 398.8816534
time: 1.20 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8663783, upper bound: 398.8663783
time: 1.39 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 5.34 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.9129977, upper bound: 398.9129977
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.9129977, upper bound: 398.9158662
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.9158662, upper bound: 398.9154052
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.9158662, upper bound: 398.9182737
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.9152035, upper bound: 398.8691338
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.9152991, upper bound: 398.8921741
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.9181313, upper bound: 398.8715492
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.9182269, upper bound: 398.8945896
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.8691338, upper bound: 398.9152035
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.8715492, upper bound: 398.9181313
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.8921741, upper bound: 398.9152991
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.8945896, upper bound: 398.9182269
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.8715256, upper bound: 398.8715359
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.8715256, upper bound: 398.8944705
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.8944602, upper bound: 398.8716315
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.8944602, upper bound: 398.8945661
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.9144023, upper bound: 398.8633344
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.9147192, upper bound: 398.8820402
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.9173301, upper bound: 398.8657498
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.9176470, upper bound: 398.8844556
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.9156067, upper bound: 398.8821731
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.9134090, upper bound: 398.8693288
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.9177291, upper bound: 398.8844032
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.9134090, upper bound: 398.8715588
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.8707347, upper bound: 398.8656307
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.8707347, upper bound: 398.8843366
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.8936693, upper bound: 398.8657263
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.8936693, upper bound: 398.8844321
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.8649391, upper bound: 398.8825553
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.8328200, upper bound: 398.8663197
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.8911499, upper bound: 398.8827268
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.8726940, upper bound: 398.8664053
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.8633344, upper bound: 398.9144023
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.8657498, upper bound: 398.9173301
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.8820402, upper bound: 398.9147192
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.8844556, upper bound: 398.9176470
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.8656307, upper bound: 398.8707347
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.8656307, upper bound: 398.8936693
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.8843366, upper bound: 398.8710516
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.8843366, upper bound: 398.8939862
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.8821731, upper bound: 398.9156067
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.8835678, upper bound: 398.9177291
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.8693288, upper bound: 398.9134090
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.8715588, upper bound: 398.9155315
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.8826879, upper bound: 398.8657021
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.8828733, upper bound: 398.8919129
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.8674129, upper bound: 398.8635723
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.8675982, upper bound: 398.8897921
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.8648295, upper bound: 398.8648295
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.8648295, upper bound: 398.8835354
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.8835354, upper bound: 398.8651464
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.8835354, upper bound: 398.8838522
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.8655892, upper bound: 398.8650312
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.8655892, upper bound: 398.8835425
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.8842951, upper bound: 398.8653481
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.8842951, upper bound: 398.8838594
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.8650312, upper bound: 398.8658989
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.8650312, upper bound: 398.8846048
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.8835425, upper bound: 398.8659449
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.8835425, upper bound: 398.8846508
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.8653824, upper bound: 398.8661006
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.8653824, upper bound: 398.8846119
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.8682573, upper bound: 398.8816534
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.34
Output dim: 0, lower bound: -398.8663783, upper bound: 398.8663783

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -143.9629669, 232.2097778, -143.9629669, 232.2097778, -376.1727295, 376.1727295
1: -158.2770081, 207.5072632, -158.2770081, 207.5072632, -365.7842407, 365.7842102
2: -157.9999237, 212.2220764, -157.9999237, 212.2220764, -370.2218933, 370.2219543
3: -186.7086945, 239.1380768, -186.7086945, 239.1380768, -425.8467407, 425.8467712
4: -160.7478485, 243.0447235, -160.7478485, 243.0447235, -403.7925720, 403.7925720

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8903976, upper bound: 398.9128082
time: 1.06 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9129071, upper bound: 398.9129063
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -143.9629669, 232.2097778, -168.5615845, 272.8850708, -416.8480225, 400.7713623
1: -158.2770081, 207.5072632, -185.7120361, 244.9842834, -403.2612915, 393.2192993
2: -157.9999237, 212.2220764, -185.3814392, 250.2069092, -408.2067566, 397.6034851
3: -186.7086945, 239.1380768, -219.6568146, 281.9783020, -468.6869812, 458.7948914
4: -160.7478485, 243.0447235, -189.2194977, 286.4551392, -447.2030029, 432.2642212

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8903976, upper bound: 398.9157368
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9129071, upper bound: 398.9158350
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -168.5615845, 272.8850708, -143.9629669, 232.2097778, -400.7713623, 416.8480225
1: -185.7120361, 244.9842834, -158.2770081, 207.5072632, -393.2192993, 403.2612915
2: -185.3814392, 250.2069092, -157.9999237, 212.2220764, -397.6035156, 408.2067566
3: -219.6568146, 281.9783020, -186.7086945, 239.1380768, -458.7948914, 468.6869507
4: -189.2194977, 286.4551392, -160.7478485, 243.0447235, -432.2642212, 447.2030029

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8891990, upper bound: 398.9151924
time: 1.24 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9158350, upper bound: 398.9153218
time: 1.26 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -168.5615845, 272.8850708, -168.5615845, 272.8850708, -441.4466553, 441.4466248
1: -185.7120361, 244.9842834, -185.7120361, 244.9842834, -430.6963196, 430.6963196
2: -185.3814392, 250.2069092, -185.3814392, 250.2069092, -435.5883484, 435.5883484
3: -219.6568146, 281.9783020, -219.6568146, 281.9783020, -501.6351318, 501.6351318
4: -189.2194977, 286.4551392, -189.2194977, 286.4551392, -475.6746216, 475.6746216

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8891990, upper bound: 398.9181210
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9158350, upper bound: 398.9182504
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -143.9629669, 232.2097778, -171.1241913, 273.7840576, -417.7470093, 403.3339844
1: -158.2770081, 207.5072632, -187.5600433, 243.4932709, -401.7702332, 395.0673218
2: -157.9999237, 212.2220764, -187.7182159, 247.8797760, -405.8796082, 399.9403076
3: -186.7086945, 239.1380768, -220.0472717, 280.7072754, -467.4159546, 459.1853027
4: -160.7478485, 243.0447235, -189.6266632, 285.1177368, -445.8655701, 432.6713867

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8926939, upper bound: 398.8690356
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8926939, upper bound: 398.8691338
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -143.9629669, 232.2097778, -181.0280762, 291.8794556, -435.8424072, 413.2378540
1: -158.2770081, 207.5072632, -198.6312103, 259.5796204, -417.8566284, 406.1383667
2: -157.9999237, 212.2220764, -198.8130798, 264.4493713, -422.4492493, 411.0350952
3: -186.7086945, 239.1380768, -233.6219635, 299.2090454, -485.9177246, 472.7600403
4: -160.7478485, 243.0447235, -201.1061249, 303.9932556, -464.7410889, 444.1508484

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8927895, upper bound: 398.8920760
time: 1.09 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8927895, upper bound: 398.8921742
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -168.5615845, 272.8850708, -170.6629181, 273.1109619, -441.6724854, 443.5479736
1: -185.7120361, 244.9842834, -187.0670471, 242.8964386, -428.6084595, 432.0513306
2: -185.3814392, 250.2069092, -187.2187500, 247.2723083, -432.6537476, 437.4256287
3: -219.6568146, 281.9783020, -219.4798279, 280.0138855, -499.6707153, 501.4581299
4: -189.2194977, 286.4551392, -189.1351776, 284.4145508, -473.6340332, 475.5903320

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8914953, upper bound: 398.8714198
time: 1.34 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8914953, upper bound: 398.8715492
time: 1.21 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -168.5615845, 272.8850708, -180.5950928, 291.2523193, -459.8138733, 453.4801636
1: -185.7120361, 244.9842834, -198.1653137, 259.0220337, -444.7340698, 443.1495056
2: -185.3814392, 250.2069092, -198.3438873, 263.8871155, -449.2684937, 448.5507812
3: -219.6568146, 281.9783020, -233.0887604, 298.5600891, -518.2169189, 515.0670166
4: -189.2194977, 286.4551392, -200.6416321, 303.3370361, -492.5565186, 487.0967712

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8915909, upper bound: 398.8944602
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8915909, upper bound: 398.8945896
time: 1.06 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -171.1241913, 273.7840576, -143.9462280, 232.1853638, -403.3095398, 417.7302856
1: -187.5600433, 243.4932709, -158.2592773, 207.4861755, -395.0462036, 401.7525024
2: -187.7182159, 247.8797760, -157.9818268, 212.2011414, -399.9193726, 405.8615723
3: -220.0472717, 280.7072754, -186.6891327, 239.1136780, -459.1608582, 467.3964233
4: -189.6266632, 285.1177368, -160.7307587, 243.0197144, -432.6463623, 445.8484802

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8674904, upper bound: 398.9152035
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8504934, upper bound: 398.9133337
time: 1.10 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8690219, upper bound: 398.9151945
time: 1.20 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -170.6629181, 273.1109619, -163.9609985, 267.3163147, -437.9792480, 437.0718994
1: -187.0670471, 242.8964386, -180.9280548, 239.9805450, -427.0476074, 423.8244934
2: -187.2187500, 247.2723083, -180.4935455, 244.8801575, -432.0988159, 427.7658691
3: -219.4798279, 280.0138855, -214.5056763, 276.2247620, -495.7045898, 494.5195618
4: -189.1351776, 284.4145508, -184.6694031, 280.3129272, -469.4481201, 469.0838928

Time for backsubstitution: 2.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8671263, upper bound: 398.9178740
time: 1.21 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8529088, upper bound: 398.9162616
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8714373, upper bound: 398.9181224
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -181.0280762, 291.8794556, -143.9462280, 232.1853638, -413.2134094, 435.8256836
1: -198.6312103, 259.5796204, -158.2592773, 207.4861755, -406.1172791, 417.8388977
2: -198.8130798, 264.4493713, -157.9818268, 212.2011414, -411.0141907, 422.4312134
3: -233.6219635, 299.2090454, -186.6891327, 239.1136780, -472.7356567, 485.8981934
4: -201.1061249, 303.9932556, -160.7307587, 243.0197144, -444.1258545, 464.7239990

Time for backsubstitution: 2.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8905308, upper bound: 398.9152938
time: 1.36 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8921408, upper bound: 398.9152991
time: 1.08 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8892357, upper bound: 398.9145695
time: 1.13 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -180.5950928, 291.2523193, -163.9609985, 267.3163147, -447.9114075, 455.2132874
1: -198.1653137, 259.0220337, -180.9280548, 239.9805450, -438.1458130, 439.9500732
2: -198.3438873, 263.8871155, -180.4935455, 244.8801575, -443.2240295, 444.3806458
3: -233.0887604, 298.5600891, -214.5056763, 276.2247620, -509.3135376, 513.0657959
4: -200.6416321, 303.3370361, -184.6694031, 280.3129272, -480.9545593, 488.0063782

Time for backsubstitution: 2.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8901666, upper bound: 398.9179644
time: 1.21 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8944410, upper bound: 398.9174809
time: 1.14 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8914812, upper bound: 398.9167514
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -171.1241913, 273.7840576, -171.1241913, 273.7840576, -444.9082642, 444.9082642
1: -187.5600433, 243.4932709, -187.5600433, 243.4932709, -431.0532837, 431.0532837
2: -187.7182159, 247.8797760, -187.7182159, 247.8797760, -435.5979919, 435.5979919
3: -220.0472717, 280.7072754, -220.0472717, 280.7072754, -500.7545471, 500.7545471
4: -189.6266632, 285.1177368, -189.6266632, 285.1177368, -474.7443542, 474.7443542

Time for backsubstitution: 2.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8527755, upper bound: 398.8696024
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8714580, upper bound: 398.8714632
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -171.1241913, 273.7840576, -181.0280762, 291.8794556, -463.0036621, 454.8121338
1: -187.5600433, 243.4932709, -198.6312103, 259.5796204, -447.1396484, 442.1243591
2: -187.7182159, 247.8797760, -198.8130798, 264.4493713, -452.1676025, 446.6928101
3: -220.0472717, 280.7072754, -233.6219635, 299.2090454, -519.2563477, 514.3292236
4: -189.6266632, 285.1177368, -201.1061249, 303.9932556, -493.6199036, 486.2238770

Time for backsubstitution: 2.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8527755, upper bound: 398.8927699
time: 1.08 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8714580, upper bound: 398.8944615
time: 1.07 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -181.0280762, 291.8794556, -171.1241913, 273.7840576, -454.8121033, 463.0036621
1: -198.6312103, 259.5796204, -187.5600433, 243.4932709, -442.1243591, 447.1396484
2: -198.8130798, 264.4493713, -187.7182159, 247.8797760, -446.6928101, 452.1676025
3: -233.6219635, 299.2090454, -220.0472717, 280.7072754, -514.3292236, 519.2563477
4: -201.1061249, 303.9932556, -189.6266632, 285.1177368, -486.2238770, 493.6199036

Time for backsubstitution: 2.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8937951, upper bound: 398.8655217
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8908266, upper bound: 398.8647922
time: 0.95 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 6.58 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.58
Output dim: 0, lower bound: -398.8903976, upper bound: 398.9128082
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.58
Output dim: 0, lower bound: -398.9129071, upper bound: 398.9129063
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.58
Output dim: 0, lower bound: -398.8903976, upper bound: 398.9157368
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.58
Output dim: 0, lower bound: -398.9129071, upper bound: 398.9158350
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.58
Output dim: 0, lower bound: -398.8891990, upper bound: 398.9151924
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.58
Output dim: 0, lower bound: -398.9158350, upper bound: 398.9153218
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.58
Output dim: 0, lower bound: -398.8891990, upper bound: 398.9181210
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.58
Output dim: 0, lower bound: -398.9158350, upper bound: 398.9182504
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.58
Output dim: 0, lower bound: -398.8926939, upper bound: 398.8690356
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.58
Output dim: 0, lower bound: -398.8926939, upper bound: 398.8691338
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.58
Output dim: 0, lower bound: -398.8927895, upper bound: 398.8920760
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.58
Output dim: 0, lower bound: -398.8927895, upper bound: 398.8921742
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.58
Output dim: 0, lower bound: -398.8914953, upper bound: 398.8714198
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.58
Output dim: 0, lower bound: -398.8914953, upper bound: 398.8715492
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.58
Output dim: 0, lower bound: -398.8915909, upper bound: 398.8944602
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.58
Output dim: 0, lower bound: -398.8915909, upper bound: 398.8945896
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.58
Output dim: 0, lower bound: -398.8504934, upper bound: 398.9133337
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.58
Output dim: 0, lower bound: -398.8690219, upper bound: 398.9151945
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.58
Output dim: 0, lower bound: -398.8529088, upper bound: 398.9162616
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.58
Output dim: 0, lower bound: -398.8714373, upper bound: 398.9181224
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.58
Output dim: 0, lower bound: -398.8921408, upper bound: 398.9152991
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.58
Output dim: 0, lower bound: -398.8892357, upper bound: 398.9145695
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.58
Output dim: 0, lower bound: -398.8944410, upper bound: 398.9174809
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.58
Output dim: 0, lower bound: -398.8914812, upper bound: 398.9167514
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.58
Output dim: 0, lower bound: -398.8527755, upper bound: 398.8696024
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.58
Output dim: 0, lower bound: -398.8714580, upper bound: 398.8714632
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.58
Output dim: 0, lower bound: -398.8527755, upper bound: 398.8927699
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.58
Output dim: 0, lower bound: -398.8714580, upper bound: 398.8944615
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.58
Output dim: 0, lower bound: -398.8937951, upper bound: 398.8655217
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.58
Output dim: 0, lower bound: -398.8908266, upper bound: 398.8647922
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.58
Output dim: 0, lower bound: -398.8944602, upper bound: 398.8945661
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.58
Output dim: 0, lower bound: -398.9144023, upper bound: 398.8633344
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.58
Output dim: 0, lower bound: -398.9147192, upper bound: 398.8820402
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.58
Output dim: 0, lower bound: -398.9173301, upper bound: 398.8657498
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.58
Output dim: 0, lower bound: -398.9176470, upper bound: 398.8844556
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.58
Output dim: 0, lower bound: -398.9156067, upper bound: 398.8821731
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.58
Output dim: 0, lower bound: -398.9134090, upper bound: 398.8693288
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.58
Output dim: 0, lower bound: -398.9177291, upper bound: 398.8844032
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.58
Output dim: 0, lower bound: -398.9134090, upper bound: 398.8715588
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.58
Output dim: 0, lower bound: -398.8707347, upper bound: 398.8656307
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.58
Output dim: 0, lower bound: -398.8707347, upper bound: 398.8843366
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.58
Output dim: 0, lower bound: -398.8936693, upper bound: 398.8657263
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.58
Output dim: 0, lower bound: -398.8936693, upper bound: 398.8844321
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.58
Output dim: 0, lower bound: -398.8649391, upper bound: 398.8825553
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.58
Output dim: 0, lower bound: -398.8328200, upper bound: 398.8663197
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.58
Output dim: 0, lower bound: -398.8911499, upper bound: 398.8827268
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.58
Output dim: 0, lower bound: -398.8726940, upper bound: 398.8664053
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.58
Output dim: 0, lower bound: -398.8633344, upper bound: 398.9144023
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.58
Output dim: 0, lower bound: -398.8657498, upper bound: 398.9173301
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.58
Output dim: 0, lower bound: -398.8820402, upper bound: 398.9147192
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.58
Output dim: 0, lower bound: -398.8844556, upper bound: 398.9176470
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.58
Output dim: 0, lower bound: -398.8656307, upper bound: 398.8707347
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.58
Output dim: 0, lower bound: -398.8656307, upper bound: 398.8936693
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.58
Output dim: 0, lower bound: -398.8843366, upper bound: 398.8710516
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.58
Output dim: 0, lower bound: -398.8843366, upper bound: 398.8939862
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.58
Output dim: 0, lower bound: -398.8821731, upper bound: 398.9156067
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.58
Output dim: 0, lower bound: -398.8835678, upper bound: 398.9177291
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.58
Output dim: 0, lower bound: -398.8693288, upper bound: 398.9134090
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.58
Output dim: 0, lower bound: -398.8715588, upper bound: 398.9155315
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.58
Output dim: 0, lower bound: -398.8826879, upper bound: 398.8657021
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.58
Output dim: 0, lower bound: -398.8828733, upper bound: 398.8919129
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.58
Output dim: 0, lower bound: -398.8674129, upper bound: 398.8635723
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.58
Output dim: 0, lower bound: -398.8675982, upper bound: 398.8897921
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.58
Output dim: 0, lower bound: -398.8648295, upper bound: 398.8648295
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.58
Output dim: 0, lower bound: -398.8648295, upper bound: 398.8835354
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.58
Output dim: 0, lower bound: -398.8835354, upper bound: 398.8651464
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.58
Output dim: 0, lower bound: -398.8835354, upper bound: 398.8838522
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.58
Output dim: 0, lower bound: -398.8655892, upper bound: 398.8650312
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.58
Output dim: 0, lower bound: -398.8655892, upper bound: 398.8835425
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.58
Output dim: 0, lower bound: -398.8842951, upper bound: 398.8653481
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.58
Output dim: 0, lower bound: -398.8842951, upper bound: 398.8838594
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.58
Output dim: 0, lower bound: -398.8650312, upper bound: 398.8658989
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.58
Output dim: 0, lower bound: -398.8650312, upper bound: 398.8846048
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.58
Output dim: 0, lower bound: -398.8835425, upper bound: 398.8659449
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.58
Output dim: 0, lower bound: -398.8835425, upper bound: 398.8846508
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.58
Output dim: 0, lower bound: -398.8653824, upper bound: 398.8661006
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.58
Output dim: 0, lower bound: -398.8653824, upper bound: 398.8846119
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.58
Output dim: 0, lower bound: -398.8682573, upper bound: 398.8816534
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.58
Output dim: 0, lower bound: -398.8663783, upper bound: 398.8663783
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=482.57733154296875
rel_dist={0: [-398.93901443925324, 398.93901443925324]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9348608, upper bound: 398.8867203
time: 1.17 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8866934, upper bound: 398.8866934
time: 1.54 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.93 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.93
Output dim: 0, lower bound: -398.9348608, upper bound: 398.8867203
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.93
Output dim: 0, lower bound: -398.8866934, upper bound: 398.8866934

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -164.8828735, 273.6366882, -179.2202911, 303.3570251, -468.2398987, 452.8569946
1: -181.5148926, 242.5971832, -197.4927673, 268.0154114, -449.5303040, 440.0898438
2: -181.4661713, 246.9006042, -197.7517548, 272.0600586, -453.5261230, 444.6523438
3: -214.6846161, 279.4453430, -234.1109924, 308.6250000, -523.3095093, 513.5563354
4: -184.8080750, 283.3780212, -201.8509827, 312.5909424, -497.3990173, 485.2290039

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8865091, upper bound: 398.8865091
time: 1.13 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8865091, upper bound: 398.8865091
time: 1.31 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -382.0864258, 576.8370972, -178.5576172, 302.3097534, -671.6203003, 750.7130127
1: -417.7886047, 527.2669067, -196.7674561, 267.0998230, -673.1426392, 718.6796265
2: -416.6779175, 536.5142822, -197.0205383, 271.1442261, -677.9602051, 728.1585083
3: -487.6743774, 608.4529419, -233.2631989, 307.5670166, -786.5380249, 835.4491577
4: -418.2083435, 614.6218872, -201.1223145, 311.5185852, -721.4415283, 813.4503174

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8865091, upper bound: 398.8866934
time: 1.06 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8865091, upper bound: 398.8866934
time: 1.02 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.76 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.76
Output dim: 0, lower bound: -398.8865091, upper bound: 398.8865091
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.76
Output dim: 0, lower bound: -398.8865091, upper bound: 398.8865091
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.76
Output dim: 0, lower bound: -398.8865091, upper bound: 398.8866934
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.76
Output dim: 0, lower bound: -398.8865091, upper bound: 398.8866934

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -164.8828735, 273.6366882, -164.8828735, 273.6366882, -438.5195618, 438.5195618
1: -181.5148926, 242.5971832, -181.5148926, 242.5971832, -424.1119995, 424.1119995
2: -181.4661713, 246.9006042, -181.4661713, 246.9006042, -428.3666992, 428.3666992
3: -214.6846161, 279.4453430, -214.6846161, 279.4453430, -494.1299438, 494.1299438
4: -184.8080750, 283.3780212, -184.8080750, 283.3780212, -468.1860962, 468.1860962

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9305913, upper bound: 398.8852436
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8943187, upper bound: 398.8851953
time: 1.16 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -164.8828735, 273.6366882, -381.9971313, 576.6977539, -737.1074219, 642.9606934
1: -181.5148926, 242.5971832, -417.6867065, 527.1360474, -703.5108643, 648.7600708
2: -181.4661713, 246.9006042, -416.5794983, 536.3850708, -712.6371460, 654.1838989
3: -214.6846161, 279.4453430, -487.5523071, 608.2994385, -816.9176636, 758.3770752
4: -184.8080750, 283.3780212, -418.1069336, 614.4724121, -797.1302490, 693.3044434

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9305913, upper bound: 398.8852436
time: 1.10 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8943187, upper bound: 398.8851953
time: 1.00 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -381.9971313, 576.6977539, -164.8828735, 273.6366882, -642.9606934, 737.1074219
1: -417.6867065, 527.1360474, -181.5148926, 242.5971832, -648.7600098, 703.5108643
2: -416.5794983, 536.3850708, -181.4661713, 246.9006042, -654.1838989, 712.6371460
3: -487.5523071, 608.2994385, -214.6846161, 279.4453430, -758.3770142, 816.9176636
4: -418.1069336, 614.4724121, -184.8080750, 283.3780212, -693.3045044, 797.1302490

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8847302, upper bound: 398.8846837
time: 0.91 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8851132, upper bound: 398.8854018
time: 1.40 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -382.1474609, 576.9548950, -382.1474609, 576.9548950, -937.3775635, 937.3775635
1: -417.8582153, 527.3723145, -417.8582153, 527.3723145, -923.8921509, 923.8921509
2: -416.7462769, 536.6198730, -416.7462769, 536.6198730, -933.7025757, 933.7026367
3: -487.7610474, 608.5759277, -487.7610474, 608.5759277, -1077.0296631, 1077.0296631
4: -418.2799072, 614.7428589, -418.2799072, 614.7428589, -1018.3641968, 1018.3641357

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8847302, upper bound: 398.8846837
time: 0.96 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8851132, upper bound: 398.8854018
time: 1.25 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.74 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.74
Output dim: 0, lower bound: -398.9305913, upper bound: 398.8852436
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.74
Output dim: 0, lower bound: -398.8943187, upper bound: 398.8851953
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.74
Output dim: 0, lower bound: -398.9305913, upper bound: 398.8852436
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.74
Output dim: 0, lower bound: -398.8943187, upper bound: 398.8851953
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.74
Output dim: 0, lower bound: -398.8847302, upper bound: 398.8846837
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.74
Output dim: 0, lower bound: -398.8851132, upper bound: 398.8854018
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.74
Output dim: 0, lower bound: -398.8847302, upper bound: 398.8846837
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.74
Output dim: 0, lower bound: -398.8851132, upper bound: 398.8854018

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -160.8784637, 265.9215088, -164.8828735, 273.6366882, -434.5151062, 430.8043823
1: -177.0686951, 235.9393463, -181.5148926, 242.5971832, -419.6657104, 417.4541626
2: -176.9501801, 240.2561340, -181.4661713, 246.9006042, -423.8507690, 421.7222900
3: -209.3154907, 271.8285217, -214.6846161, 279.4453430, -488.7608337, 486.5131226
4: -180.1105499, 275.6749573, -184.8080750, 283.3780212, -463.4885864, 460.4830322

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8944298, upper bound: 398.8944298
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8944298, upper bound: 398.8944298
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -182.5338745, 294.6768799, -163.9465637, 271.9315796, -454.4654541, 458.6234436
1: -200.3133240, 262.0391235, -180.4746704, 241.1051178, -441.4184570, 442.5137024
2: -200.4873199, 266.9307861, -180.4207153, 245.4162140, -445.9035339, 447.3514709
3: -235.6646729, 302.0430603, -213.4435272, 277.7338257, -513.3984985, 515.4865112
4: -202.8659515, 306.8235168, -183.7133484, 281.6534119, -484.5193176, 490.5368652

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8944298, upper bound: 398.8944373
time: 1.26 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8944298, upper bound: 398.8944372
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -160.8784637, 265.9215088, -381.9358521, 576.6179199, -733.1184692, 635.1378174
1: -177.0686951, 235.9393463, -417.6166992, 527.0575562, -699.0517578, 642.0987549
2: -176.9501801, 240.2561340, -416.5125732, 536.3084717, -708.1061401, 647.5204468
3: -209.3154907, 271.8285217, -487.4707336, 608.2067871, -811.5011597, 750.6167603
4: -180.1105499, 275.6749573, -418.0388184, 614.3827515, -792.3787842, 685.5056152

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8936853, upper bound: 398.8847721
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8936853, upper bound: 398.8851953
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -182.5338745, 294.6768799, -380.7490845, 574.5186157, -752.1338501, 662.4402466
1: -200.3133240, 262.0391235, -416.3102722, 525.2225952, -719.9884033, 667.3371582
2: -200.4873199, 266.9307861, -415.1892395, 534.4572144, -729.4296265, 673.2014160
3: -235.6646729, 302.0430603, -485.9092102, 606.1108398, -835.4205933, 779.7608032
4: -202.8659515, 306.8235168, -416.6757507, 612.2615967, -812.5882568, 715.3579102

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8936232, upper bound: 398.8835275
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8919706, upper bound: 398.8709600
time: 1.16 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -379.3073120, 571.3442383, -164.8828735, 273.6366882, -639.9810181, 731.4604492
1: -414.6747437, 522.7958984, -181.5148926, 242.5971832, -645.4816895, 698.9234009
2: -413.5457153, 532.0236206, -181.4661713, 246.9006042, -650.8913574, 708.0319214
3: -483.8923340, 603.3825684, -214.6846161, 279.4453430, -754.4989624, 811.7159424
4: -414.9602661, 609.4316406, -184.8080750, 283.3780212, -689.9499512, 791.8882446

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8847721, upper bound: 398.8936853
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8847721, upper bound: 398.8936853
time: 1.14 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -404.0063782, 606.5186768, -163.9465637, 271.9315796, -660.4965820, 763.6580811
1: -441.4174805, 555.4885254, -180.4746704, 241.1051178, -668.4694824, 728.5595703
2: -440.3652344, 565.4312744, -180.4207153, 245.4162140, -674.0253906, 738.4300537
3: -514.5853271, 641.2687378, -213.4435272, 277.7338257, -781.7115479, 845.9978027
4: -441.0548401, 647.4913940, -183.7133484, 281.6534119, -712.6112061, 827.5217285

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8851953, upper bound: 398.8946761
time: 0.93 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8851953, upper bound: 398.8946761
time: 1.07 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -379.4693604, 571.5990601, -382.1474609, 576.9548950, -934.4069214, 931.7293091
1: -414.8598328, 523.0347900, -417.8582153, 527.3723145, -920.6241455, 919.3082886
2: -413.7246399, 532.2597656, -416.7462769, 536.6198730, -930.4193115, 929.0993652
3: -484.1145325, 603.6630249, -487.7610474, 608.5759277, -1073.1617432, 1071.8328857
4: -415.1448059, 609.7048950, -418.2799072, 614.7428589, -1015.0192871, 1013.1253662

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8844989, upper bound: 398.8844989
time: 1.23 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8844989, upper bound: 398.8846837
time: 1.02 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -404.3253174, 607.0194092, -380.7558594, 574.5313721, -954.2674561, 963.6591797
1: -441.7807312, 555.9595947, -416.3179626, 525.2340088, -943.0526123, 948.6262207
2: -440.7174683, 565.8958130, -415.1967468, 534.4687500, -952.9715576, 959.1748047
3: -515.0227661, 641.8205566, -485.9187012, 606.1242676, -1099.7730713, 1105.7348633
4: -441.4168091, 648.0268555, -416.6835938, 612.2749023, -1037.0500488, 1048.4859619

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8716511, upper bound: 398.8823076
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8697401, upper bound: 398.8697401
time: 0.80 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.08 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.08
Output dim: 0, lower bound: -398.8944298, upper bound: 398.8944298
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.08
Output dim: 0, lower bound: -398.8944298, upper bound: 398.8944298
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.08
Output dim: 0, lower bound: -398.8944298, upper bound: 398.8944373
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.08
Output dim: 0, lower bound: -398.8944298, upper bound: 398.8944372
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.08
Output dim: 0, lower bound: -398.8936853, upper bound: 398.8847721
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.08
Output dim: 0, lower bound: -398.8936853, upper bound: 398.8851953
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.08
Output dim: 0, lower bound: -398.8936232, upper bound: 398.8835275
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.08
Output dim: 0, lower bound: -398.8919706, upper bound: 398.8709600
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.08
Output dim: 0, lower bound: -398.8847721, upper bound: 398.8936853
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.08
Output dim: 0, lower bound: -398.8847721, upper bound: 398.8936853
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.08
Output dim: 0, lower bound: -398.8851953, upper bound: 398.8946761
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.08
Output dim: 0, lower bound: -398.8851953, upper bound: 398.8946761
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.08
Output dim: 0, lower bound: -398.8844989, upper bound: 398.8844989
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.08
Output dim: 0, lower bound: -398.8844989, upper bound: 398.8846837
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.08
Output dim: 0, lower bound: -398.8716511, upper bound: 398.8823076
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.08
Output dim: 0, lower bound: -398.8697401, upper bound: 398.8697401

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -160.8784637, 265.9215088, -160.8784637, 265.9215088, -426.7999268, 426.7999268
1: -177.0686951, 235.9393463, -177.0686951, 235.9393463, -413.0078430, 413.0078430
2: -176.9501801, 240.2561340, -176.9501801, 240.2561340, -417.2062988, 417.2062988
3: -209.3154907, 271.8285217, -209.3154907, 271.8285217, -481.1440125, 481.1440125
4: -180.1105499, 275.6749573, -180.1105499, 275.6749573, -455.7854919, 455.7854919

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9135001, upper bound: 398.8913021
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9159796, upper bound: 398.8940914
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -160.8784637, 265.9215088, -182.5338745, 294.6768799, -455.5553284, 448.4553528
1: -177.0686951, 235.9393463, -200.3133240, 262.0391235, -439.1076965, 436.2526245
2: -176.9501801, 240.2561340, -200.4873199, 266.9307861, -443.8809814, 440.7434692
3: -209.3154907, 271.8285217, -235.6646729, 302.0430603, -511.3585510, 507.4931946
4: -180.1105499, 275.6749573, -202.8659515, 306.8235168, -486.9340515, 478.5408325

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9135001, upper bound: 398.8913021
time: 1.04 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9159796, upper bound: 398.8940914
time: 1.06 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -182.5338745, 294.6768799, -160.8635864, 265.9017334, -448.4355469, 455.5404663
1: -200.3133240, 262.0391235, -177.0530396, 235.9219971, -436.2352905, 439.0921631
2: -200.4873199, 266.9307861, -176.9342804, 240.2380371, -440.7253418, 443.8650513
3: -235.6646729, 302.0430603, -209.2984161, 271.8085022, -507.4731750, 511.3414917
4: -202.8659515, 306.8235168, -180.0955811, 275.6540222, -478.5199585, 486.9190369

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8716207, upper bound: 398.8922527
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8923588, upper bound: 398.8923589
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -182.5338745, 294.6768799, -182.5338745, 294.6768799, -477.2107544, 477.2107544
1: -200.3133240, 262.0391235, -200.3133240, 262.0391235, -462.3524475, 462.3524475
2: -200.4873199, 266.9307861, -200.4873199, 266.9307861, -467.4180908, 467.4180908
3: -235.6646729, 302.0430603, -235.6646729, 302.0430603, -537.7077026, 537.7077026
4: -202.8659515, 306.8235168, -202.8659515, 306.8235168, -509.6894226, 509.6894226

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8716207, upper bound: 398.8922527
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8923588, upper bound: 398.8923589
time: 1.21 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -160.8784637, 265.9215088, -379.2485657, 571.2697144, -727.4765015, 632.1608276
1: -177.0686951, 235.9393463, -414.6075134, 522.7219238, -694.4686279, 638.8236694
2: -176.9501801, 240.2561340, -413.4816284, 531.9515991, -703.5054932, 644.2306519
3: -209.3154907, 271.8285217, -483.8143921, 603.2953491, -806.3046265, 746.7422485
4: -180.1105499, 275.6749573, -414.8950500, 609.3471069, -787.1417236, 682.1539307

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9120721, upper bound: 398.8824699
time: 1.06 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9158178, upper bound: 398.8849466
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -160.8784637, 265.9215088, -403.8773499, 606.3755493, -760.5303955, 654.3101196
1: -177.0686951, 235.9393463, -441.2706604, 555.3403931, -725.0647583, 663.2172852
2: -176.9501801, 240.2561340, -440.2252808, 565.2892456, -734.8696289, 668.7647705
3: -209.3154907, 271.8285217, -514.4169312, 641.0933838, -841.7345581, 775.5958862
4: -180.1105499, 275.6749573, -440.9140320, 647.3239746, -823.7875977, 706.4662476

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9120721, upper bound: 398.8827719
time: 1.19 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9158178, upper bound: 398.8852436
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -180.4615021, 291.5382690, -367.3287659, 552.6267700, -727.5932617, 645.1300659
1: -198.0119629, 259.1118469, -401.3681335, 505.0691223, -696.9521484, 648.7140503
2: -198.2005157, 263.9687500, -400.3193359, 514.1419678, -706.2626953, 654.6383057
3: -232.9667816, 298.6684570, -468.1655273, 583.0934448, -809.0850830, 758.0352173
4: -200.5160217, 303.4109497, -401.3240662, 588.9522095, -786.4198608, 696.0340576

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8657021, upper bound: 398.8813872
time: 1.13 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8908761, upper bound: 398.8815842
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -182.5338745, 294.6768799, -377.5008240, 568.8563843, -746.4122925, 659.1416016
1: -200.3133240, 262.0391235, -412.7144775, 520.1802368, -714.9139404, 663.6947632
2: -200.4873199, 266.9307861, -411.5745239, 529.3690186, -724.3311157, 669.5442505
3: -235.6646729, 302.0430603, -481.6205750, 600.3592529, -829.6016235, 775.4649048
4: -202.8659515, 306.8235168, -412.9826660, 606.4819946, -806.8351440, 711.6281738

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8635723, upper bound: 398.8674129
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8897035, upper bound: 398.8675982
time: 1.10 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -379.2485657, 571.2697144, -160.8784637, 265.9215088, -632.1607666, 727.4765015
1: -414.6075134, 522.7219238, -177.0686951, 235.9393463, -638.8236694, 694.4685669
2: -413.4816284, 531.9515991, -176.9501801, 240.2561340, -644.2306519, 703.5054932
3: -483.8143921, 603.2953491, -209.3154907, 271.8285217, -746.7422485, 806.3046265
4: -414.8950500, 609.3471069, -180.1105499, 275.6749573, -682.1539307, 787.1417236

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8820616, upper bound: 398.8927635
time: 1.02 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8657208, upper bound: 398.8910276
time: 1.04 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8837492, upper bound: 398.8915042
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -379.4634705, 571.5874634, -182.5338745, 294.6768799, -660.8963013, 748.9848022
1: -414.8529663, 523.0245361, -200.3133240, 262.0391235, -665.6454468, 717.6085205
2: -413.7179260, 532.2493286, -200.4873199, 266.9307861, -671.4989624, 727.0429688
3: -484.1060791, 603.6509399, -235.6646729, 302.0430603, -777.7703857, 832.7521973
4: -415.1377869, 609.6930542, -202.8659515, 306.8235168, -713.6395874, 809.8715820

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8820616, upper bound: 398.8927635
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8657208, upper bound: 398.8910276
time: 0.97 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8837492, upper bound: 398.8915042
time: 1.13 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -403.9550476, 606.4494019, -160.8635864, 265.9017334, -654.3583374, 760.5897217
1: -441.3590088, 555.4207764, -177.0530396, 235.9219971, -663.2773438, 725.1293945
2: -440.3089294, 565.3652954, -176.9342804, 240.2380371, -668.8215332, 734.9301147
3: -514.5166626, 641.1892090, -209.2984161, 271.8085022, -775.6644287, 841.8131714
4: -440.9976196, 647.4147339, -180.0955811, 275.6540222, -706.5227661, 823.8631592

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8835275, upper bound: 398.8936232
time: 0.98 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8709600, upper bound: 398.8919706
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -404.2296448, 606.8332520, -182.5338745, 294.6768799, -683.1668701, 782.1762695
1: -441.6716309, 555.7925415, -200.3133240, 262.0391235, -690.1765747, 748.3526001
2: -440.6102295, 565.7286987, -200.4873199, 266.9307861, -696.1672974, 758.5473633
3: -514.8862915, 641.6262817, -235.6646729, 302.0430603, -806.7827148, 868.3574219
4: -441.3048706, 647.8365479, -202.8659515, 306.8235168, -738.0889893, 846.6835938

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8835275, upper bound: 398.8936232
time: 0.87 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8709600, upper bound: 398.8919706
time: 1.28 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -379.4693604, 571.5990601, -379.4693604, 571.5990601, -928.7586060, 928.7586060
1: -414.8598328, 523.0347900, -414.8598328, 523.0347900, -916.0402222, 916.0402222
2: -413.7246399, 532.2597656, -413.7246399, 532.2597656, -925.8160400, 925.8160400
3: -484.1145325, 603.6630249, -484.1145325, 603.6630249, -1067.9649658, 1067.9648438
4: -415.1448059, 609.7048950, -415.1448059, 609.7048950, -1009.7804565, 1009.7804565

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8651464, upper bound: 398.8831377
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8835891, upper bound: 398.8834626
time: 1.22 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -379.4693604, 571.5990601, -404.3253174, 607.0194092, -962.1144409, 951.1172485
1: -414.8598328, 523.0347900, -441.7807312, 555.9595947, -946.9337158, 940.6716919
2: -413.7246399, 532.2597656, -440.7174683, 565.8958130, -957.4715576, 950.5836792
3: -484.1145325, 603.6630249, -515.0227661, 641.8205566, -1103.7432861, 1097.1033936
4: -415.1448059, 609.7048950, -441.4168091, 648.0268555, -1046.7667236, 1034.3321533

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8651464, upper bound: 398.8831377
time: 1.29 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8835891, upper bound: 398.8834626
time: 1.26 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -401.9616089, 603.4229736, -367.3545837, 552.6772461, -929.3765259, 945.9121094
1: -439.1440735, 552.5398560, -401.3975525, 505.1145630, -919.6650391, 929.5646362
2: -438.1141968, 562.4509888, -400.3482971, 514.1878052, -929.4149780, 940.1844482
3: -511.9191284, 637.8927612, -468.2027893, 583.1458740, -1073.0129395, 1083.5102539
4: -438.7531433, 644.0626831, -401.3546143, 589.0045776, -1010.5630493, 1028.6553955

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8697401, upper bound: 398.8697401
time: 1.23 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8697401, upper bound: 398.8697401
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -404.3253174, 607.0194092, -377.5113220, 568.8766479, -948.5530396, 960.3640747
1: -441.7807312, 555.9595947, -412.7264709, 520.1984863, -937.9845581, 944.9881592
2: -440.7174683, 565.8958130, -411.5863037, 529.3872681, -947.8796387, 955.5217285
3: -515.0227661, 641.8205566, -481.6356201, 600.3805542, -1093.9616699, 1101.4445801
4: -441.4168091, 648.0268555, -412.9949646, 606.5029297, -1031.3044434, 1044.7607422

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8697401, upper bound: 398.8697401
time: 1.02 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8697401, upper bound: 398.8697401
time: 0.88 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.48 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -398.9135001, upper bound: 398.8913021
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -398.9159796, upper bound: 398.8940914
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -398.9135001, upper bound: 398.8913021
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -398.9159796, upper bound: 398.8940914
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -398.8716207, upper bound: 398.8922527
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -398.8923588, upper bound: 398.8923589
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -398.8716207, upper bound: 398.8922527
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -398.8923588, upper bound: 398.8923589
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -398.9120721, upper bound: 398.8824699
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -398.9158178, upper bound: 398.8849466
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -398.9120721, upper bound: 398.8827719
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -398.9158178, upper bound: 398.8852436
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -398.8657021, upper bound: 398.8813872
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -398.8908761, upper bound: 398.8815842
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -398.8635723, upper bound: 398.8674129
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -398.8897035, upper bound: 398.8675982
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -398.8657208, upper bound: 398.8910276
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -398.8837492, upper bound: 398.8915042
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -398.8657208, upper bound: 398.8910276
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -398.8837492, upper bound: 398.8915042
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -398.8835275, upper bound: 398.8936232
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -398.8709600, upper bound: 398.8919706
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -398.8835275, upper bound: 398.8936232
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -398.8709600, upper bound: 398.8919706
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -398.8651464, upper bound: 398.8831377
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -398.8835891, upper bound: 398.8834626
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -398.8651464, upper bound: 398.8831377
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -398.8835891, upper bound: 398.8834626
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -398.8697401, upper bound: 398.8697401
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -398.8697401, upper bound: 398.8697401
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -398.8697401, upper bound: 398.8697401
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -398.8697401, upper bound: 398.8697401

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -143.9629669, 232.2097778, -158.9890442, 262.1518555, -406.1148071, 391.1988220
1: -158.2770081, 207.5072632, -174.9698334, 232.7444763, -391.0214539, 382.4771118
2: -157.9999237, 212.2220764, -174.8262939, 237.0706024, -395.0704346, 387.0483398
3: -186.7086945, 239.1380768, -206.7827911, 268.1541748, -454.8628540, 445.9208679
4: -160.7478485, 243.0447235, -177.9195862, 272.0112305, -432.7590942, 420.9642944

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9119667, upper bound: 398.9119667
time: 1.48 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9119667, upper bound: 398.9133601
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -168.5615845, 272.8850708, -159.0193329, 263.0093384, -431.5708923, 431.9043579
1: -185.7120361, 244.9842834, -175.0561371, 233.4058533, -419.1178894, 420.0404053
2: -185.3814392, 250.2069092, -174.9178467, 237.7013397, -423.0827332, 425.1247253
3: -219.6568146, 281.9783020, -206.9951019, 268.9091492, -488.5659485, 488.9733887
4: -189.2194977, 286.4551392, -178.1210327, 272.7068481, -461.9263306, 464.5761719

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9141425, upper bound: 398.8916069
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9165144, upper bound: 398.9165144
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -143.9629669, 232.2097778, -180.7624054, 291.0589905, -435.0219727, 412.9721680
1: -158.2770081, 207.5072632, -198.3411102, 259.0017090, -417.2787170, 405.8483887
2: -157.9999237, 212.2220764, -198.4929810, 263.8899536, -421.8897705, 410.7150574
3: -186.7086945, 239.1380768, -233.2714233, 298.5535889, -485.2622070, 472.4094849
4: -160.7478485, 243.0447235, -200.8250427, 303.3454590, -464.0932922, 443.8697510

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9128235, upper bound: 398.8691299
time: 1.21 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9127452, upper bound: 398.8895747
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -168.5615845, 272.8850708, -180.1689911, 291.1474304, -459.7090149, 453.0540771
1: -185.7120361, 244.9842834, -197.7626038, 258.9192505, -444.6312561, 442.7468872
2: -185.3814392, 250.2069092, -197.9082642, 263.7772522, -449.1586914, 448.1151733
3: -219.6568146, 281.9783020, -232.7270813, 298.4205933, -518.0773926, 514.7053833
4: -189.2194977, 286.4551392, -200.3198547, 303.1471252, -492.3666077, 486.7749939

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9158100, upper bound: 398.8715424
time: 2.59 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9159176, upper bound: 398.8924074
time: 1.08 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -171.1241913, 273.7840576, -160.4503479, 265.1492310, -436.2733765, 434.2343445
1: -187.5600433, 243.4932709, -176.5890961, 235.2496490, -422.8096924, 420.0823669
2: -187.7182159, 247.8797760, -176.4758453, 239.5427551, -427.2609863, 424.3556213
3: -220.0472717, 280.7072754, -208.7257538, 271.0246277, -491.0718079, 489.4330139
4: -189.6266632, 285.1177368, -179.6088867, 274.8669434, -464.4935608, 464.7265930

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8691299, upper bound: 398.9128235
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8715424, upper bound: 398.9158100
time: 1.26 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -181.0280762, 291.8794556, -160.8635864, 265.9017334, -446.9297791, 452.7430420
1: -198.6312103, 259.5796204, -177.0530396, 235.9219971, -434.5531311, 436.6326599
2: -198.8130798, 264.4493713, -176.9342804, 240.2380371, -439.0510864, 441.3836670
3: -233.6219635, 299.2090454, -209.2984161, 271.8085022, -505.4304810, 508.5074463
4: -201.1061249, 303.9932556, -180.0955811, 275.6540222, -476.7601318, 484.0888062

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8895747, upper bound: 398.9127452
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8924074, upper bound: 398.9159176
time: 1.17 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -171.1241913, 273.7840576, -182.1033478, 293.9021606, -465.0263672, 455.8873901
1: -187.5600433, 243.4932709, -199.8294067, 261.3673401, -448.9273682, 443.3226929
2: -187.7182159, 247.8797760, -200.0099945, 266.2268677, -453.9450684, 447.8897400
3: -220.0472717, 280.7072754, -235.0736237, 301.2584229, -521.3056641, 515.7808838
4: -189.6266632, 285.1177368, -202.3710785, 306.0362854, -495.6629639, 487.4887695

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8715239, upper bound: 398.8715359
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8715239, upper bound: 398.8922527
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -181.0280762, 291.8794556, -182.5338745, 294.6768799, -475.7049561, 474.4133301
1: -198.6312103, 259.5796204, -200.3133240, 262.0391235, -460.6702271, 459.8929443
2: -198.8130798, 264.4493713, -200.4873199, 266.9307861, -465.7437744, 464.9367065
3: -233.6219635, 299.2090454, -235.6646729, 302.0430603, -535.6648560, 534.8737183
4: -201.1061249, 303.9932556, -202.8659515, 306.8235168, -507.9296265, 506.8591919

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8922008, upper bound: 398.8716207
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8922008, upper bound: 398.8923589
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -143.9629669, 232.2097778, -377.6483765, 568.2493286, -707.8604126, 597.6677246
1: -158.2770081, 207.5072632, -412.8017883, 520.2257690, -673.5430908, 610.2257080
2: -157.9999237, 212.2220764, -411.6896362, 529.4492188, -682.2933350, 615.6470337
3: -186.7086945, 239.1380768, -481.6412964, 600.4543457, -781.1489868, 713.6190796
4: -160.7478485, 243.0447235, -413.0361633, 606.4602051, -765.1525269, 648.9653931

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9120690, upper bound: 398.8809687
time: 1.15 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9105383, upper bound: 398.8633273
time: 1.09 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9109004, upper bound: 398.8814437
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -168.5615845, 272.8850708, -376.9451904, 567.3020630, -731.1719360, 637.8541260
1: -185.7120361, 244.9842834, -412.0506897, 519.2943115, -699.5383301, 645.5397949
2: -185.3814392, 250.2069092, -410.9062500, 528.4774780, -708.4508057, 651.3334351
3: -219.6568146, 281.9783020, -480.7610474, 599.3175049, -812.2966309, 754.4536743
4: -189.2194977, 286.4551392, -412.2957458, 605.3366699, -791.7507935, 690.6325073

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9149463, upper bound: 398.8827054
time: 1.35 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9152817, upper bound: 398.8657460
time: 1.16 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9157619, upper bound: 398.8839477
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -143.9629669, 232.2097778, -402.2935181, 603.3978882, -740.9522705, 619.8247070
1: -158.2770081, 207.5072632, -439.4830933, 552.8911133, -704.1835327, 634.6277466
2: -157.9999237, 212.2220764, -438.4531555, 562.8328247, -713.6994629, 640.1938477
3: -186.7086945, 239.1380768, -512.2592773, 638.2955322, -816.6224976, 742.4787598
4: -160.7478485, 243.0447235, -439.0690613, 644.4635010, -801.8209839, 673.2864380

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9124954, upper bound: 398.8817856
time: 1.20 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9104378, upper bound: 398.8693254
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -168.5615845, 272.8850708, -401.6253052, 602.5734253, -764.3941650, 660.0593262
1: -185.7120361, 244.9842834, -438.7790833, 552.0300293, -730.2525635, 670.0026855
2: -185.3814392, 250.2069092, -437.7124939, 561.9342651, -739.9345703, 675.9353027
3: -219.6568146, 281.9783020, -511.4497070, 637.2512207, -847.8638306, 783.3925171
4: -189.2194977, 286.4551392, -438.3943481, 643.4540405, -828.5386963, 715.0264893

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9155136, upper bound: 398.8840599
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9108759, upper bound: 398.8715588
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -168.9836121, 270.4949341, -366.4844360, 551.1549072, -714.6561279, 622.2285767
1: -185.1853485, 240.4412689, -400.4185486, 503.7494202, -682.6755371, 628.4067383
2: -185.3573303, 244.7700195, -399.3745422, 512.8094482, -692.0081177, 633.7022705
3: -217.2583923, 277.1882629, -467.0028992, 581.5702515, -791.6542358, 734.5826416
4: -187.1987152, 281.5633545, -400.3417664, 587.4116211, -771.2398682, 672.7971802

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8649391, upper bound: 398.8811657
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8649391, upper bound: 398.8812956
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -178.9494629, 288.7316589, -367.3067627, 552.5836182, -726.0436401, 642.3818970
1: -196.3197479, 256.6437988, -401.3429871, 505.0304871, -695.2277222, 646.2856445
2: -196.5198975, 261.4756165, -400.2945251, 514.1028442, -704.5286865, 652.1983032
3: -230.9078522, 295.8279419, -468.1336670, 583.0487061, -806.9955444, 755.2385864
4: -198.7490082, 300.5632935, -401.2979431, 588.9072266, -784.6376343, 693.2558594

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8899611, upper bound: 398.8813511
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8899611, upper bound: 398.8814795
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -171.1241913, 273.7840576, -376.8981934, 567.7974243, -733.9968262, 636.6494751
1: -187.5600433, 243.4932709, -412.0372314, 519.2344360, -701.1069336, 643.8018188
2: -187.7182159, 247.8797760, -410.9041138, 528.4152832, -710.5895386, 649.0237427
3: -220.0472717, 280.7072754, -480.7937317, 599.2695312, -812.7148438, 752.5196533
4: -189.6266632, 285.1177368, -412.2831421, 605.3843384, -792.1993408, 688.8071289

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8628173, upper bound: 398.8653442
time: 1.13 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8632608, upper bound: 398.8663187
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -181.0280762, 291.8794556, -377.4818726, 568.8196411, -744.8815308, 656.4063110
1: -198.6312103, 259.5796204, -412.6929016, 520.1473999, -713.2064209, 661.2789917
2: -198.8130798, 264.4493713, -411.5533447, 529.3360596, -722.6096802, 667.1204224
3: -233.6219635, 299.2090454, -481.5935974, 600.3210449, -827.5360107, 772.6799927
4: -201.1061249, 303.9932556, -412.9605103, 606.4444580, -805.0680542, 708.8723755

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8874301, upper bound: 398.8655294
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8878736, upper bound: 398.8665040
time: 1.00 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -368.8053894, 551.9740601, -160.4651947, 265.1690063, -620.3764038, 706.5750122
1: -402.8101196, 505.6078796, -176.6046906, 235.2669525, -625.9372559, 675.8891602
2: -401.7810974, 514.5733032, -176.4917297, 239.5608673, -631.2558594, 684.5772095
3: -469.2990112, 583.7431641, -208.7427979, 271.0446167, -731.0239868, 784.9697266
4: -402.4276428, 589.3398438, -179.6239319, 274.8877258, -668.3901367, 765.8264160

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8633273, upper bound: 398.9105383
time: 1.17 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8657460, upper bound: 398.9152817
time: 2.46 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -377.8862305, 568.6209717, -160.8784637, 265.9215088, -630.6720581, 724.8255615
1: -413.0809937, 520.4889526, -177.0686951, 235.9393463, -637.2105103, 692.2025757
2: -411.9556580, 529.6585693, -176.9501801, 240.2561340, -642.5650024, 701.1987915
3: -481.9333801, 600.7451782, -209.3154907, 271.8285217, -744.7832031, 803.7189331
4: -413.3024597, 606.7645874, -180.1105499, 275.6749573, -680.4990234, 784.5269775

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8814437, upper bound: 398.9109004
time: 1.00 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8839477, upper bound: 398.9157619
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -369.0470276, 552.3290405, -182.1033478, 293.9021606, -649.1311035, 728.1317139
1: -403.0852051, 505.9463806, -199.8294067, 261.3673401, -652.7958374, 699.0635376
2: -402.0468140, 514.9060059, -200.0099945, 266.2268677, -658.5589600, 708.1288452
3: -469.6240845, 584.1408081, -235.0736237, 301.2584229, -762.1027832, 811.4548950
4: -402.6987915, 589.7242432, -202.3710785, 306.0362854, -699.9122314, 788.5960693

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8656307, upper bound: 398.8707347
time: 0.99 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8656307, upper bound: 398.8910276
time: 0.99 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -378.1191711, 568.9608154, -182.5338745, 294.6768799, -659.4238892, 746.3554077
1: -413.3469238, 520.8134766, -200.3133240, 262.0391235, -664.0508423, 715.3641357
2: -412.2115784, 529.9777832, -200.4873199, 266.9307861, -669.8514404, 724.7574463
3: -482.2487183, 601.1270752, -235.6646729, 302.0430603, -775.8326416, 830.1923828
4: -413.5648499, 607.1357422, -202.8659515, 306.8235168, -712.0031738, 807.2819824

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8836575, upper bound: 398.8710516
time: 0.99 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8836575, upper bound: 398.8915042
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -392.4074707, 587.5618286, -159.0656738, 263.1517944, -639.2406616, 739.4120483
1: -428.4705811, 538.2255859, -175.0447998, 233.3592377, -647.0744019, 705.4581909
2: -427.4966736, 548.0471802, -174.9463348, 237.6276398, -652.6281128, 715.1315918
3: -499.2272339, 621.5214233, -206.9264069, 268.8572693, -756.8768921, 819.2805786
4: -427.7804260, 627.4034424, -178.0682983, 272.6578979, -689.7964478, 801.3775635

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8817856, upper bound: 398.9124954
time: 1.28 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8840599, upper bound: 398.9155136
time: 0.85 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -400.6930847, 600.8972778, -160.8635864, 265.9017334, -651.0443115, 754.9566650
1: -437.7517090, 550.5214844, -177.0530396, 235.9219971, -659.6260986, 720.1874390
2: -436.6783447, 560.4235840, -176.9342804, 240.2380371, -665.1501465, 729.9596558
3: -510.2201538, 635.5836792, -209.2984161, 271.8085022, -771.3481445, 836.1311646
4: -437.3178101, 641.7301025, -180.0955811, 275.6540222, -702.8067017, 818.2070923

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8693254, upper bound: 398.9104378
time: 0.93 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8715588, upper bound: 398.9108759
time: 1.13 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -392.7612610, 588.0579224, -180.4615021, 291.5382690, -667.7213745, 760.8026733
1: -428.8677673, 538.7013550, -198.0119629, 259.1118469, -673.6654663, 728.4691772
2: -427.8847351, 548.5167847, -198.2005157, 263.9687500, -679.6727905, 738.5656128
3: -499.6963196, 622.0775757, -232.9667816, 298.6684570, -787.6085815, 845.6037598
4: -428.1711426, 627.9496460, -200.5160217, 303.4109497, -720.9970093, 823.9935303

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8813872, upper bound: 398.8657021
time: 1.42 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8815842, upper bound: 398.8908761
time: 1.27 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -401.1927795, 601.5297241, -182.5338745, 294.6768799, -680.0547485, 776.7898560
1: -438.2965698, 551.1254272, -200.3133240, 262.0391235, -686.7387695, 743.6417847
2: -437.2202759, 561.0217285, -200.4873199, 266.9307861, -692.7149048, 753.8109741
3: -510.8418884, 636.2901001, -235.6646729, 302.0430603, -802.7057495, 862.9437256
4: -437.8387756, 642.4287109, -202.8659515, 306.8235168, -734.5744019, 841.3032227

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8674129, upper bound: 398.8635723
time: 1.18 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8675982, upper bound: 398.8897035
time: 0.95 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -369.0663452, 552.3657227, -379.0773010, 570.9066162, -917.1435547, 907.9522095
1: -403.1070862, 505.9794922, -414.4213562, 522.4230957, -903.3250732, 897.5861206
2: -402.0683594, 514.9389648, -413.2884521, 531.6391602, -913.0238037, 906.9535522
3: -469.6511230, 584.1793823, -483.5800781, 602.9638672, -1052.4576416, 1046.7678223
4: -402.7211609, 589.7617798, -414.6867981, 608.9913940, -996.1712646, 988.5954590

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8648295, upper bound: 398.8648295
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8648295, upper bound: 398.8832722
time: 1.08 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -378.1610107, 569.0414429, -379.4693604, 571.5990601, -927.3214111, 926.1946411
1: -413.3947754, 520.8857422, -414.8598328, 523.0347900, -914.4857178, 913.8551636
2: -412.2584534, 530.0501709, -413.7246399, 532.2597656, -924.2080688, 923.5906372
3: -482.3082886, 601.2113037, -484.1145325, 603.6630249, -1066.0776367, 1065.4739990
4: -413.6139221, 607.2186279, -415.1448059, 609.7048950, -1008.1848755, 1007.2605591

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8832722, upper bound: 398.8651464
time: 1.24 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8832722, upper bound: 398.8835891
time: 1.37 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -369.0663452, 552.3657227, -403.9292297, 606.3278809, -950.5050049, 930.3065186
1: -403.1070862, 505.9794922, -441.3367004, 555.3476562, -934.2228394, 922.2126465
2: -402.0683594, 514.9389648, -440.2778625, 565.2761841, -944.6837769, 931.7168579
3: -469.6511230, 584.1793823, -514.4828491, 641.1206665, -1088.2426758, 1075.8985596
4: -402.7211609, 589.7617798, -440.9532166, 647.3106079, -1033.1597900, 1013.1420898

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8655867, upper bound: 398.8650312
time: 0.98 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8655867, upper bound: 398.8831377
time: 1.00 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -378.1610107, 569.0414429, -404.3253174, 607.0194092, -960.6771240, 948.5532837
1: -413.3947754, 520.8857422, -441.7807312, 555.9595947, -945.3793335, 938.4866333
2: -412.2584534, 530.0501709, -440.7174683, 565.8958130, -955.8635864, 948.3582764
3: -482.3082886, 601.2113037, -515.0227661, 641.8205566, -1101.8562012, 1094.6125488
4: -413.6139221, 607.2186279, -441.4168091, 648.0268555, -1045.1713867, 1031.8121338

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8836128, upper bound: 398.8653481
time: 1.31 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8836128, upper bound: 398.8834626
time: 1.29 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -392.9279175, 588.3497314, -367.3545837, 552.6772461, -919.7874756, 930.5037842
1: -429.0569458, 538.9698486, -401.3975525, 505.1145630, -909.1049805, 915.7352295
2: -428.0704956, 548.7850342, -400.3482971, 514.1878052, -918.9081421, 926.2437744
3: -499.9305420, 622.3893433, -468.2027893, 583.1458740, -1060.7043457, 1067.7117920
4: -428.3645935, 628.2581787, -401.3546143, 589.0045776, -999.9259644, 1012.6342163

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8711069, upper bound: 398.8808417
time: 1.02 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8711069, upper bound: 398.8823076
time: 0.95 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -401.4359131, 601.8801880, -367.3545837, 552.6772461, -928.6705933, 944.2364502
1: -438.5582275, 551.4457397, -401.3975525, 505.1145630, -918.9074097, 928.3581543
2: -437.4852295, 561.3435669, -400.3482971, 514.1878052, -928.6290894, 938.9703979
3: -511.1441040, 636.6628418, -468.2027893, 583.1458740, -1072.1252441, 1082.1425781
4: -438.0916748, 642.8013306, -401.3546143, 589.0045776, -1009.7952881, 1027.3403320

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8711069, upper bound: 398.8808417
time: 0.97 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8711069, upper bound: 398.8823076
time: 1.05 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -392.9279175, 588.3497314, -377.5113220, 568.8766479, -936.2546387, 941.0001221
1: -429.0569458, 538.9698486, -412.7264709, 520.1984863, -924.4309082, 927.3768921
2: -428.0704956, 548.7850342, -411.5863037, 529.3872681, -934.3700562, 937.7567749
3: -499.9305420, 622.3893433, -481.6356201, 600.3805542, -1078.2104492, 1081.3879395
4: -428.3645935, 628.2581787, -412.9949646, 606.5029297, -1017.6411743, 1024.4519043

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8675308, upper bound: 398.8658359
time: 1.01 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8682223, upper bound: 398.8682223
time: 1.02 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -401.4359131, 601.8801880, -377.5113220, 568.8766479, -945.5732422, 955.1403198
1: -438.5582275, 551.4457397, -412.7264709, 520.1984863, -934.6873169, 940.4295654
2: -437.4852295, 561.3435669, -411.5863037, 529.3872681, -944.5710449, 950.9392700
3: -511.1441040, 636.6628418, -481.6356201, 600.3805542, -1090.0413818, 1096.2082520
4: -438.0916748, 642.8013306, -412.9949646, 606.5029297, -1027.9227295, 1039.5620117

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8675308, upper bound: 398.8658359
time: 1.31 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8682223, upper bound: 398.8682223
time: 0.97 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 8.74 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.9119667, upper bound: 398.9119667
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.9119667, upper bound: 398.9133601
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.9141425, upper bound: 398.8916069
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.9165144, upper bound: 398.9165144
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.9128235, upper bound: 398.8691299
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.9127452, upper bound: 398.8895747
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.9158100, upper bound: 398.8715424
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.9159176, upper bound: 398.8924074
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.8691299, upper bound: 398.9128235
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.8715424, upper bound: 398.9158100
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.8895747, upper bound: 398.9127452
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.8924074, upper bound: 398.9159176
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.8715239, upper bound: 398.8715359
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.8715239, upper bound: 398.8922527
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.8922008, upper bound: 398.8716207
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.8922008, upper bound: 398.8923589
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.9105383, upper bound: 398.8633273
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.9109004, upper bound: 398.8814437
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.9152817, upper bound: 398.8657460
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.9157619, upper bound: 398.8839477
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.9124954, upper bound: 398.8817856
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.9104378, upper bound: 398.8693254
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.9155136, upper bound: 398.8840599
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.9108759, upper bound: 398.8715588
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.8649391, upper bound: 398.8811657
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.8649391, upper bound: 398.8812956
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.8899611, upper bound: 398.8813511
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.8899611, upper bound: 398.8814795
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.8628173, upper bound: 398.8653442
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.8632608, upper bound: 398.8663187
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.8874301, upper bound: 398.8655294
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.8878736, upper bound: 398.8665040
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.8633273, upper bound: 398.9105383
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.8657460, upper bound: 398.9152817
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.8814437, upper bound: 398.9109004
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.8839477, upper bound: 398.9157619
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.8656307, upper bound: 398.8707347
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.8656307, upper bound: 398.8910276
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.8836575, upper bound: 398.8710516
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.8836575, upper bound: 398.8915042
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.8817856, upper bound: 398.9124954
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.8840599, upper bound: 398.9155136
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.8693254, upper bound: 398.9104378
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.8715588, upper bound: 398.9108759
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.8813872, upper bound: 398.8657021
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.8815842, upper bound: 398.8908761
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.8674129, upper bound: 398.8635723
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.8675982, upper bound: 398.8897035
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.8648295, upper bound: 398.8648295
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.8648295, upper bound: 398.8832722
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.8832722, upper bound: 398.8651464
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.8832722, upper bound: 398.8835891
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.8655867, upper bound: 398.8650312
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.8655867, upper bound: 398.8831377
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.8836128, upper bound: 398.8653481
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.8836128, upper bound: 398.8834626
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.8711069, upper bound: 398.8808417
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.8711069, upper bound: 398.8823076
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.8711069, upper bound: 398.8808417
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.8711069, upper bound: 398.8823076
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.8675308, upper bound: 398.8658359
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.8682223, upper bound: 398.8682223
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.8675308, upper bound: 398.8658359
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 0, lower bound: -398.8682223, upper bound: 398.8682223

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -143.9629669, 232.2097778, -143.9629669, 232.2097778, -376.1727295, 376.1727295
1: -158.2770081, 207.5072632, -158.2770081, 207.5072632, -365.7842407, 365.7842102
2: -157.9999237, 212.2220764, -157.9999237, 212.2220764, -370.2218933, 370.2219543
3: -186.7086945, 239.1380768, -186.7086945, 239.1380768, -425.8467407, 425.8467712
4: -160.7478485, 243.0447235, -160.7478485, 243.0447235, -403.7925720, 403.7925720

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8903751, upper bound: 398.9115381
time: 1.14 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9119943, upper bound: 398.9116840
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -143.9629669, 232.2097778, -168.5615845, 272.8850708, -416.8480225, 400.7713623
1: -158.2770081, 207.5072632, -185.7120361, 244.9842834, -403.2612915, 393.2192993
2: -157.9999237, 212.2220764, -185.3814392, 250.2069092, -408.2067566, 397.6034851
3: -186.7086945, 239.1380768, -219.6568146, 281.9783020, -468.6869812, 458.7948914
4: -160.7478485, 243.0447235, -189.2194977, 286.4551392, -447.2030029, 432.2642212

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8903751, upper bound: 398.9128851
time: 1.30 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9119943, upper bound: 398.9132290
time: 1.31 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -168.1272278, 272.1029053, -147.8040619, 241.4996948, -409.6269226, 419.9069824
1: -185.2256470, 244.2870789, -162.5021667, 214.4191589, -399.6448059, 406.7892151
2: -184.8992310, 249.4918518, -162.3721619, 218.0658569, -402.9650879, 411.8640137
3: -219.0518951, 281.1779175, -191.4060364, 247.0608673, -466.1127625, 472.5839539
4: -188.7071228, 285.6425476, -165.0509949, 250.5380249, -439.2451172, 450.6935425

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8914832, upper bound: 398.8914832
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8914832, upper bound: 398.8916069
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -168.5615845, 272.8850708, -157.4771271, 260.2063599, -428.7678528, 430.3621826
1: -185.7120361, 244.9842834, -173.3419342, 230.9534454, -416.6654663, 418.3262024
2: -185.3814392, 250.2069092, -173.2222290, 235.2655182, -420.6469421, 423.4291382
3: -219.6568146, 281.9783020, -204.9498596, 266.1046143, -485.7614136, 486.9281616
4: -189.2194977, 286.4551392, -176.3513184, 269.9061279, -459.1256104, 462.8064575

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8916069, upper bound: 398.9162911
time: 1.39 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8916069, upper bound: 398.9165144
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -143.5522003, 231.4368744, -169.4140167, 270.2663574, -413.8185425, 400.8508911
1: -157.8149719, 206.8174591, -185.6542816, 240.5475769, -398.3625488, 392.4717407
2: -157.5424957, 211.5184021, -185.7907257, 245.0922546, -402.6347046, 397.3091431
3: -186.1294403, 238.3451233, -217.7339935, 277.3276978, -463.4570923, 456.0791016
4: -160.2596283, 242.2535248, -187.6661835, 281.7393799, -441.9990234, 429.9196777

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8926937, upper bound: 398.8690328
time: 1.44 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8926937, upper bound: 398.8691299
time: 1.17 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -143.9629669, 232.2097778, -179.2558289, 288.2536011, -432.2165222, 411.4656067
1: -158.2770081, 207.5072632, -196.6573792, 256.5357361, -414.8126526, 404.1645813
2: -157.9999237, 212.2220764, -196.8179321, 261.4505310, -419.4504395, 409.0399475
3: -186.7086945, 239.1380768, -231.2277069, 295.7113647, -482.4200439, 470.3657837
4: -160.7478485, 243.0447235, -199.0616455, 300.5094604, -461.2573242, 442.1063843

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8927753, upper bound: 398.8892913
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8927753, upper bound: 398.8895748
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -168.1272278, 272.1029053, -168.5122528, 269.9977722, -438.1249390, 440.6151733
1: -185.2256470, 244.2870789, -184.7624969, 240.1163483, -425.3419800, 429.0495605
2: -184.8992310, 249.4918518, -184.8845367, 244.4466553, -429.3458252, 434.3763733
3: -219.0518951, 281.1779175, -216.8243713, 276.7847900, -495.8366394, 498.0022888
4: -188.7071228, 285.6425476, -186.8360748, 281.1416626, -469.8487549, 472.4786377

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8914953, upper bound: 398.8714183
time: 1.45 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8914953, upper bound: 398.8715424
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -168.5615845, 272.8850708, -178.6499481, 288.3379517, -456.8995056, 451.5349426
1: -185.7120361, 244.9842834, -196.0662842, 256.4462585, -442.1582947, 441.0505676
2: -185.3814392, 250.2069092, -196.2195435, 261.2854309, -446.6668701, 446.4264526
3: -219.6568146, 281.9783020, -230.6748505, 295.5706482, -515.2274780, 512.6531372
4: -189.2194977, 286.4551392, -198.5427704, 300.3071289, -489.5265808, 484.9979248

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8915797, upper bound: 398.8921952
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8915797, upper bound: 398.8924074
time: 1.21 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -169.4140167, 270.2663574, -143.5354156, 231.4124603, -400.8264160, 413.8017578
1: -185.6542816, 240.5475769, -157.7972260, 206.7964020, -392.4506836, 398.3447876
2: -185.7907257, 245.0922546, -157.5243683, 211.4974213, -397.2881470, 402.6166077
3: -217.7339935, 277.3276978, -186.1098480, 238.3206940, -456.0546875, 463.4375305
4: -187.6661835, 281.7393799, -160.2425079, 242.2284851, -429.8946228, 441.9818726

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8674904, upper bound: 398.9128235
time: 1.06 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8503951, upper bound: 398.9093859
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8690172, upper bound: 398.9128228
time: 1.05 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -168.5122528, 269.9977722, -163.5225983, 266.5266418, -435.0388489, 433.5202332
1: -184.7624969, 240.1163483, -180.4353485, 239.2807922, -424.0432739, 420.5516968
2: -184.8845367, 244.4466553, -180.0074768, 244.1603851, -429.0449219, 424.4540405
3: -216.8243713, 276.7847900, -213.8890533, 275.4229431, -492.2473145, 490.6737671
4: -186.8360748, 281.1416626, -184.1531525, 279.4920349, -466.3281250, 465.2947693

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8665603, upper bound: 398.9144557
time: 1.11 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8528660, upper bound: 398.9152429
time: 1.23 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8714301, upper bound: 398.9157705
time: 1.11 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -179.2558289, 288.2536011, -143.9462280, 232.1853638, -411.4411621, 432.1997681
1: -196.6573792, 256.5357361, -158.2592773, 207.4861755, -404.1435242, 414.7949524
2: -196.8179321, 261.4505310, -157.9818268, 212.2011414, -409.0190430, 419.4323730
3: -231.2277069, 295.7113647, -186.6891327, 239.1136780, -470.3413696, 482.4005127
4: -199.0616455, 300.5094604, -160.7307587, 243.0197144, -442.0813599, 461.2402344

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8881972, upper bound: 398.9127452
time: 1.09 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8895747, upper bound: 398.9127395
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8872561, upper bound: 398.9122389
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -178.6499481, 288.3379517, -163.9609985, 267.3163147, -445.9661560, 452.2988586
1: -196.0662842, 256.4462585, -180.9280548, 239.9805450, -436.0468140, 437.3743286
2: -196.2195435, 261.2854309, -180.4935455, 244.8801575, -441.0997009, 441.7789917
3: -230.6748505, 295.5706482, -214.5056763, 276.2247620, -506.8995972, 510.0763245
4: -198.5427704, 300.3071289, -184.6694031, 280.3129272, -478.8557129, 484.9764099

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8880795, upper bound: 398.9146976
time: 1.11 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8923431, upper bound: 398.9157216
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8900047, upper bound: 398.9126085
time: 1.06 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -171.1241913, 273.7840576, -171.1241913, 273.7840576, -444.9082642, 444.9082642
1: -187.5600433, 243.4932709, -187.5600433, 243.4932709, -431.0532837, 431.0532837
2: -187.7182159, 247.8797760, -187.7182159, 247.8797760, -435.5979919, 435.5979919
3: -220.0472717, 280.7072754, -220.0472717, 280.7072754, -500.7545471, 500.7545471
4: -189.6266632, 285.1177368, -189.6266632, 285.1177368, -474.7443542, 474.7443542

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8526411, upper bound: 398.8696024
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8714580, upper bound: 398.8714632
time: 1.05 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -171.1241913, 273.7840576, -181.0280762, 291.8794556, -463.0036621, 454.8121338
1: -187.5600433, 243.4932709, -198.6312103, 259.5796204, -447.1396484, 442.1243591
2: -187.7182159, 247.8797760, -198.8130798, 264.4493713, -452.1676025, 446.6928101
3: -220.0472717, 280.7072754, -233.6219635, 299.2090454, -519.2563477, 514.3292236
4: -189.6266632, 285.1177368, -201.1061249, 303.9932556, -493.6199036, 486.2238770

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8526411, upper bound: 398.8909756
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8714580, upper bound: 398.8922527
time: 1.08 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -181.0280762, 291.8794556, -171.1241913, 273.7840576, -454.8121033, 463.0036621
1: -198.6312103, 259.5796204, -187.5600433, 243.4932709, -442.1243591, 447.1396484
2: -198.8130798, 264.4493713, -187.7182159, 247.8797760, -446.6928101, 452.1676025
3: -233.6219635, 299.2090454, -220.0472717, 280.7072754, -514.3292236, 519.2563477
4: -201.1061249, 303.9932556, -189.6266632, 285.1177368, -486.2238770, 493.6199036

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=482.57733154296875
rel_dist={0: [-398.9373570313671, 398.9373570313671]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9232315, upper bound: 398.8852357
time: 1.12 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8851378, upper bound: 398.8851378
time: 1.30 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.61 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.61
Output dim: 0, lower bound: -398.9232315, upper bound: 398.8852357
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.61
Output dim: 0, lower bound: -398.8851378, upper bound: 398.8851378

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -164.8828735, 273.6366882, -179.2202911, 303.3570251, -468.2398987, 452.8569946
1: -181.5148926, 242.5971832, -197.4927673, 268.0154114, -449.5303040, 440.0898438
2: -181.4661713, 246.9006042, -197.7517548, 272.0600586, -453.5261230, 444.6523438
3: -214.6846161, 279.4453430, -234.1109924, 308.6250000, -523.3095093, 513.5563354
4: -184.8080750, 283.3780212, -201.8509827, 312.5909424, -497.3990173, 485.2290039

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8849102, upper bound: 398.8849102
time: 1.25 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8849102, upper bound: 398.8849102
time: 1.04 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -382.0378418, 576.7455444, -178.1441345, 301.5676880, -670.8328247, 750.2157593
1: -417.7334290, 527.1847534, -196.3163605, 266.4815674, -672.4734497, 718.1522217
2: -416.6237793, 536.4317627, -196.5529327, 270.5289307, -677.2954712, 727.6116943
3: -487.6061096, 608.3573608, -232.7352142, 306.8548584, -785.7624512, 834.8328247
4: -418.1519470, 614.5274048, -200.6661987, 310.7998352, -720.6711426, 812.9034424

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8849102, upper bound: 398.8851378
time: 1.02 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8849102, upper bound: 398.8851378
time: 0.92 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.42 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.42
Output dim: 0, lower bound: -398.8849102, upper bound: 398.8849102
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.42
Output dim: 0, lower bound: -398.8849102, upper bound: 398.8849102
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.42
Output dim: 0, lower bound: -398.8849102, upper bound: 398.8851378
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.42
Output dim: 0, lower bound: -398.8849102, upper bound: 398.8851378

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -164.8828735, 273.6366882, -164.8828735, 273.6366882, -438.5195618, 438.5195618
1: -181.5148926, 242.5971832, -181.5148926, 242.5971832, -424.1119995, 424.1119995
2: -181.4661713, 246.9006042, -181.4661713, 246.9006042, -428.3666992, 428.3666992
3: -214.6846161, 279.4453430, -214.6846161, 279.4453430, -494.1299438, 494.1299438
4: -184.8080750, 283.3780212, -184.8080750, 283.3780212, -468.1860962, 468.1860962

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9192678, upper bound: 398.8839014
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8929162, upper bound: 398.8836897
time: 1.13 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -164.8828735, 273.6366882, -381.8911133, 576.5324707, -736.9484253, 642.8603516
1: -181.5148926, 242.5971832, -417.5708313, 526.9865112, -703.3659058, 648.6494751
2: -181.4661713, 246.9006042, -416.4637146, 536.2330933, -712.4892578, 654.0733643
3: -214.6846161, 279.4453430, -487.4162598, 608.1259155, -816.7497559, 758.2451172
4: -184.8080750, 283.3780212, -417.9928894, 614.2974243, -796.9583740, 693.1943359

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9192678, upper bound: 398.8839014
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8929162, upper bound: 398.8836897
time: 1.01 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -381.8911133, 576.5324707, -164.8828735, 273.6366882, -642.8604126, 736.9483643
1: -417.5708313, 526.9865112, -181.5148926, 242.5971832, -648.6494751, 703.3659058
2: -416.4637146, 536.2330322, -181.4661713, 246.9006042, -654.0733643, 712.4891357
3: -487.4162598, 608.1259155, -214.6846161, 279.4453430, -758.2451172, 816.7497559
4: -417.9928894, 614.2974243, -184.8080750, 283.3780212, -693.1943359, 796.9583740

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8831975, upper bound: 398.8831779
time: 1.54 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8835996, upper bound: 398.8839227
time: 1.10 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -382.1474609, 576.9548950, -382.1474609, 576.9548950, -937.3775635, 937.3775635
1: -417.8582153, 527.3723145, -417.8582153, 527.3723145, -923.8921509, 923.8921509
2: -416.7462769, 536.6198730, -416.7462769, 536.6198730, -933.7025757, 933.7026367
3: -487.7610474, 608.5759277, -487.7610474, 608.5759277, -1077.0296631, 1077.0296631
4: -418.2799072, 614.7428589, -418.2799072, 614.7428589, -1018.3641968, 1018.3641357

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8831975, upper bound: 398.8831779
time: 7.53 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8835996, upper bound: 398.8839227
time: 1.15 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 11.18 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 11.18
Output dim: 0, lower bound: -398.9192678, upper bound: 398.8839014
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 11.18
Output dim: 0, lower bound: -398.8929162, upper bound: 398.8836897
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 11.18
Output dim: 0, lower bound: -398.9192678, upper bound: 398.8839014
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 11.18
Output dim: 0, lower bound: -398.8929162, upper bound: 398.8836897
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 11.18
Output dim: 0, lower bound: -398.8831975, upper bound: 398.8831779
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 11.18
Output dim: 0, lower bound: -398.8835996, upper bound: 398.8839227
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 11.18
Output dim: 0, lower bound: -398.8831975, upper bound: 398.8831779
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 11.18
Output dim: 0, lower bound: -398.8835996, upper bound: 398.8839227

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -160.8784637, 265.9215088, -164.8828735, 273.6366882, -434.5151062, 430.8043823
1: -177.0686951, 235.9393463, -181.5148926, 242.5971832, -419.6657104, 417.4541626
2: -176.9501801, 240.2561340, -181.4661713, 246.9006042, -423.8507690, 421.7222900
3: -209.3154907, 271.8285217, -214.6846161, 279.4453430, -488.7608337, 486.5131226
4: -180.1105499, 275.6749573, -184.8080750, 283.3780212, -463.4885864, 460.4830322

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8930971, upper bound: 398.8930971
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8930971, upper bound: 398.8930971
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -182.5338745, 294.6768799, -163.2853241, 270.6809387, -453.2147522, 457.9622192
1: -200.3133240, 262.0391235, -179.7385101, 240.0155792, -440.3288879, 441.7776184
2: -200.4873199, 266.9307861, -179.6765137, 244.3400726, -444.8273926, 446.6072693
3: -235.6646729, 302.0430603, -212.5583191, 276.4852295, -512.1499023, 514.6013794
4: -202.8659515, 306.8235168, -182.9321747, 280.3990173, -483.2649536, 489.7556763

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8930971, upper bound: 398.8930971
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8930971, upper bound: 398.8930971
time: 1.27 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -160.8784637, 265.9215088, -381.7951050, 576.4121704, -732.9194946, 635.0062256
1: -177.0686951, 235.9393463, -417.4645386, 526.8709717, -698.8701172, 641.9549561
2: -176.9501801, 240.2561340, -416.3594666, 536.1190186, -707.9211426, 647.3756714
3: -209.3154907, 271.8285217, -487.2951660, 607.9901123, -811.2904663, 750.4473267
4: -180.1105499, 275.6749573, -417.8912048, 614.1635742, -792.1630859, 685.3639526

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8920438, upper bound: 398.8832468
time: 1.12 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8920438, upper bound: 398.8836897
time: 1.33 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -182.5338745, 294.6768799, -379.5971069, 572.5003662, -750.0614014, 661.2620850
1: -200.3133240, 262.0391235, -415.0315857, 523.4321899, -718.1508789, 666.0318604
2: -200.4873199, 266.9307861, -413.9064941, 532.6550293, -727.5802002, 671.8954468
3: -235.6646729, 302.0430603, -484.3737488, 604.0565796, -833.3110962, 778.1996460
4: -202.8659515, 306.8235168, -415.3451538, 610.1963501, -810.4838257, 714.0054932

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8922058, upper bound: 398.8812696
time: 1.17 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8916350, upper bound: 398.8708467
time: 1.10 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -379.2091064, 571.1932983, -164.8828735, 273.6366882, -639.8883667, 731.3151245
1: -414.5674744, 522.6590576, -181.5148926, 242.5971832, -645.3797607, 698.7907104
2: -413.4384155, 531.8844604, -181.4661713, 246.9006042, -650.7892456, 707.8966064
3: -483.7668457, 603.2237549, -214.6846161, 279.4453430, -754.3773193, 811.5623779
4: -414.8551025, 609.2708740, -184.8080750, 283.3780212, -689.8485107, 791.7304077

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8832468, upper bound: 398.8920438
time: 1.12 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8832468, upper bound: 398.8920438
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -403.7027588, 606.1366577, -163.2853241, 270.6809387, -658.9793091, 762.6326294
1: -441.0960693, 555.1368408, -179.7385101, 240.0155792, -667.0981445, 727.4885864
2: -440.0384216, 565.0750122, -179.6765137, 244.3400726, -672.6610107, 737.3402100
3: -514.2259521, 640.8605347, -212.5583191, 276.4852295, -780.1168823, 844.7199707
4: -440.7526245, 647.0750732, -182.9321747, 280.3990173, -711.0764771, 826.3325806

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8836897, upper bound: 398.8932828
time: 1.18 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8836897, upper bound: 398.8932828
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -379.4693604, 571.5990601, -382.1474609, 576.9548950, -934.4069214, 931.7293091
1: -414.8598328, 523.0347900, -417.8582153, 527.3723145, -920.6241455, 919.3082886
2: -413.7246399, 532.2597656, -416.7462769, 536.6198730, -930.4193115, 929.0993652
3: -484.1145325, 603.6630249, -487.7610474, 608.5759277, -1073.1617432, 1071.8328857
4: -415.1448059, 609.7048950, -418.2799072, 614.7428589, -1015.0192871, 1013.1253662

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8830233, upper bound: 398.8830233
time: 1.19 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8830233, upper bound: 398.8831779
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -404.3253174, 607.0194092, -379.6565857, 572.6085205, -952.2858887, 962.5320435
1: -441.7807312, 555.9595947, -415.0988464, 523.5296631, -941.2976685, 947.3788452
2: -440.7174683, 565.8958130, -413.9725342, 532.7536011, -951.2063599, 957.9259033
3: -515.0227661, 641.8205566, -484.4561157, 604.1697998, -1097.7591553, 1104.2451172
4: -441.4168091, 648.0268555, -415.4136658, 610.3090210, -1035.0432129, 1047.1925049

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8709638, upper bound: 398.8799586
time: 0.93 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8696133, upper bound: 398.8696133
time: 0.83 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.32 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.32
Output dim: 0, lower bound: -398.8930971, upper bound: 398.8930971
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.32
Output dim: 0, lower bound: -398.8930971, upper bound: 398.8930971
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.32
Output dim: 0, lower bound: -398.8930971, upper bound: 398.8930971
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.32
Output dim: 0, lower bound: -398.8930971, upper bound: 398.8930971
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.32
Output dim: 0, lower bound: -398.8920438, upper bound: 398.8832468
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.32
Output dim: 0, lower bound: -398.8920438, upper bound: 398.8836897
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.32
Output dim: 0, lower bound: -398.8922058, upper bound: 398.8812696
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.32
Output dim: 0, lower bound: -398.8916350, upper bound: 398.8708467
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.32
Output dim: 0, lower bound: -398.8832468, upper bound: 398.8920438
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.32
Output dim: 0, lower bound: -398.8832468, upper bound: 398.8920438
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.32
Output dim: 0, lower bound: -398.8836897, upper bound: 398.8932828
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.32
Output dim: 0, lower bound: -398.8836897, upper bound: 398.8932828
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.32
Output dim: 0, lower bound: -398.8830233, upper bound: 398.8830233
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.32
Output dim: 0, lower bound: -398.8830233, upper bound: 398.8831779
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.32
Output dim: 0, lower bound: -398.8709638, upper bound: 398.8799586
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.32
Output dim: 0, lower bound: -398.8696133, upper bound: 398.8696133

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -160.8784637, 265.9215088, -160.8784637, 265.9215088, -426.7999268, 426.7999268
1: -177.0686951, 235.9393463, -177.0686951, 235.9393463, -413.0078430, 413.0078430
2: -176.9501801, 240.2561340, -176.9501801, 240.2561340, -417.2062988, 417.2062988
3: -209.3154907, 271.8285217, -209.3154907, 271.8285217, -481.1440125, 481.1440125
4: -180.1105499, 275.6749573, -180.1105499, 275.6749573, -455.7854919, 455.7854919

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9007908, upper bound: 398.8825220
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9118598, upper bound: 398.8927529
time: 1.42 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -160.8784637, 265.9215088, -182.5338745, 294.6768799, -455.5553284, 448.4553528
1: -177.0686951, 235.9393463, -200.3133240, 262.0391235, -439.1076965, 436.2526245
2: -176.9501801, 240.2561340, -200.4873199, 266.9307861, -443.8809814, 440.7434692
3: -209.3154907, 271.8285217, -235.6646729, 302.0430603, -511.3585510, 507.4931946
4: -180.1105499, 275.6749573, -202.8659515, 306.8235168, -486.9340515, 478.5408325

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9007908, upper bound: 398.8825220
time: 1.07 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9118598, upper bound: 398.8927529
time: 1.12 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -182.5338745, 294.6768799, -160.8635864, 265.9017334, -448.4355469, 455.5404663
1: -200.3133240, 262.0391235, -177.0530396, 235.9219971, -436.2352905, 439.0921631
2: -200.4873199, 266.9307861, -176.9342804, 240.2380371, -440.7253418, 443.8650513
3: -235.6646729, 302.0430603, -209.2984161, 271.8085022, -507.4731750, 511.3414917
4: -202.8659515, 306.8235168, -180.0955811, 275.6540222, -478.5199585, 486.9190369

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8714421, upper bound: 398.8905354
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8909853, upper bound: 398.8909853
time: 1.23 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -182.5338745, 294.6768799, -182.5338745, 294.6768799, -477.2107544, 477.2107544
1: -200.3133240, 262.0391235, -200.3133240, 262.0391235, -462.3524475, 462.3524475
2: -200.4873199, 266.9307861, -200.4873199, 266.9307861, -467.4180908, 467.4180908
3: -235.6646729, 302.0430603, -235.6646729, 302.0430603, -537.7077026, 537.7077026
4: -202.8659515, 306.8235168, -202.8659515, 306.8235168, -509.6894226, 509.6894226

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8714421, upper bound: 398.8905354
time: 1.07 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8909853, upper bound: 398.8909853
time: 1.20 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -160.8784637, 265.9215088, -379.1163940, 571.0793457, -727.2922974, 632.0374756
1: -177.0686951, 235.9393463, -414.4649658, 522.5488892, -694.3002930, 638.6889038
2: -176.9501801, 240.2561340, -413.3379517, 531.7756348, -703.3338013, 644.0951538
3: -209.3154907, 271.8285217, -483.6500854, 603.0941772, -806.1091309, 746.5841675
4: -180.1105499, 275.6749573, -414.7571106, 609.1430664, -786.9407349, 682.0218506

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8933698, upper bound: 398.8823622
time: 1.25 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9180232, upper bound: 398.8825575
time: 1.21 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -160.8784637, 265.9215088, -403.5485840, 605.9624634, -760.1263428, 654.0108643
1: -177.0686951, 235.9393463, -440.9240723, 554.9619751, -724.6929932, 662.8958130
2: -176.9501801, 240.2561340, -439.8716736, 564.9049072, -734.4913330, 668.4381104
3: -209.3154907, 271.8285217, -514.0304565, 640.6540527, -841.3029785, 775.2291260
4: -180.1105499, 275.6749573, -440.5889282, 646.8748779, -823.3430786, 706.1577148

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8933698, upper bound: 398.8823676
time: 1.34 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9180232, upper bound: 398.8825660
time: 1.23 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -178.4749756, 288.5193787, -366.2376404, 550.8674927, -723.8276367, 640.9367065
1: -195.8151398, 256.3054810, -400.1760864, 503.5355835, -693.1640015, 644.5955200
2: -196.0048676, 261.1285706, -399.1148682, 512.5960083, -702.5026855, 650.4533691
3: -230.3947601, 295.4327393, -466.7634583, 581.3311768, -804.6981201, 753.2869263
4: -198.2784729, 300.1458435, -400.1191406, 587.1558838, -782.3487549, 691.4486694

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8654150, upper bound: 398.8784810
time: 1.32 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8895473, upper bound: 398.8787749
time: 1.30 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -181.0868835, 292.0321960, -375.7873230, 566.2148438, -742.2604980, 654.7822876
1: -198.6988068, 259.6244202, -410.8584290, 517.8161621, -710.8641357, 659.4328003
2: -198.8712311, 264.4786682, -409.6916809, 526.9861450, -720.2880859, 665.2306519
3: -233.7211304, 299.2853699, -479.4542542, 597.6370850, -824.8568115, 770.5298462
4: -201.1918793, 304.0563049, -411.1198120, 603.7286377, -802.3428955, 707.0677490

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8635442, upper bound: 398.8673649
time: 1.65 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8891407, upper bound: 398.8675628
time: 1.35 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -379.1163940, 571.0793457, -160.8784637, 265.9215088, -632.0374756, 727.2922974
1: -414.4649658, 522.5488892, -177.0686951, 235.9393463, -638.6889038, 694.3002930
2: -413.3379517, 531.7756348, -176.9501801, 240.2561340, -644.0951538, 703.3338623
3: -483.6500854, 603.0941772, -209.3154907, 271.8285217, -746.5841064, 806.1091309
4: -414.7571106, 609.1430664, -180.1105499, 275.6749573, -682.0218506, 786.9407349

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8800410, upper bound: 398.8906827
time: 1.99 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8655450, upper bound: 398.8886404
time: 1.27 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8822457, upper bound: 398.8898623
time: 1.31 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -379.4143677, 571.4991455, -182.5338745, 294.6768799, -660.8486938, 748.9006348
1: -414.7976990, 522.9447632, -200.3133240, 262.0391235, -665.5917969, 717.5317993
2: -413.6635437, 532.1687012, -200.4873199, 266.9307861, -671.4461670, 726.9648438
3: -484.0384521, 603.5582886, -235.6646729, 302.0430603, -777.7039795, 832.6635132
4: -415.0814209, 609.6007080, -202.8659515, 306.8235168, -713.5846558, 809.7811279

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8800410, upper bound: 398.8906827
time: 1.34 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8655450, upper bound: 398.8886404
time: 1.17 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8822457, upper bound: 398.8898623
time: 1.24 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -403.6234741, 606.0336304, -160.8635864, 265.9017334, -654.0565796, 760.1829834
1: -441.0092468, 555.0393677, -177.0530396, 235.9219971, -662.9528809, 724.7546997
2: -439.9523926, 564.9781494, -176.9342804, 240.2380371, -668.4921875, 734.5489502
3: -514.1264648, 640.7462158, -209.2984161, 271.8085022, -775.2943726, 841.3780518
4: -440.6694641, 646.9620361, -180.0955811, 275.6540222, -706.2113647, 823.4152832

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8812696, upper bound: 398.8922058
time: 1.28 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8708467, upper bound: 398.8916350
time: 1.36 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -404.1325684, 606.6862793, -182.5338745, 294.6768799, -683.0758667, 782.0349121
1: -441.5658569, 555.6586304, -200.3133240, 262.0391235, -690.0762329, 748.2228394
2: -440.5044250, 565.5928955, -200.4873199, 266.9307861, -696.0667725, 758.4152222
3: -514.7628784, 641.4710083, -235.6646729, 302.0430603, -806.6636963, 868.2073364
4: -441.2013550, 647.6801758, -202.8659515, 306.8235168, -737.9897461, 846.5299072

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8812696, upper bound: 398.8922058
time: 1.47 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8708467, upper bound: 398.8916350
time: 1.18 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -379.4693604, 571.5990601, -379.4693604, 571.5990601, -928.7586060, 928.7586060
1: -414.8598328, 523.0347900, -414.8598328, 523.0347900, -916.0402222, 916.0402222
2: -413.7246399, 532.2597656, -413.7246399, 532.2597656, -925.8160400, 925.8160400
3: -484.1145325, 603.6630249, -484.1145325, 603.6630249, -1067.9649658, 1067.9648438
4: -415.1448059, 609.7048950, -415.1448059, 609.7048950, -1009.7804565, 1009.7804565

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8650748, upper bound: 398.8814831
time: 1.26 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8820453, upper bound: 398.8819320
time: 1.25 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -379.4693604, 571.5990601, -404.3253174, 607.0194092, -962.1144409, 951.1172485
1: -414.8598328, 523.0347900, -441.7807312, 555.9595947, -946.9337158, 940.6716919
2: -413.7246399, 532.2597656, -440.7174683, 565.8958130, -957.4715576, 950.5836792
3: -484.1145325, 603.6630249, -515.0227661, 641.8205566, -1103.7432861, 1097.1033936
4: -415.1448059, 609.7048950, -441.4168091, 648.0268555, -1046.7667236, 1034.3321533

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8650748, upper bound: 398.8814831
time: 1.40 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8820453, upper bound: 398.8819320
time: 1.30 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -399.6912537, 599.9772949, -366.5277405, 551.2548828, -925.6367798, 941.5725098
1: -436.6146545, 549.2716675, -400.4858398, 503.8895264, -915.8801880, 925.3253784
2: -435.6145935, 559.1546021, -399.4289856, 512.9545288, -925.6489868, 935.8982544
3: -508.9430542, 634.1348877, -467.1152649, 581.7420044, -1068.5910645, 1078.6145020
4: -436.1995850, 640.2662354, -400.4136353, 587.5757446, -1006.5486450, 1023.8590088

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8696133, upper bound: 398.8696133
time: 1.21 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8696133, upper bound: 398.8696133
time: 1.05 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -403.1186218, 604.8264160, -376.4185791, 566.9642334, -945.3430176, 956.9903564
1: -440.4370422, 554.0007935, -411.5209351, 518.5050659, -934.8585815, 941.7460327
2: -439.3658142, 563.9218140, -410.3694153, 527.6834106, -944.7406006, 952.2638550
3: -513.3930054, 639.5822754, -480.1843567, 598.4395142, -1090.3005371, 1097.6572266
4: -440.0088501, 645.7865601, -411.7347717, 604.5515137, -1027.8869629, 1041.2385254

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8696133, upper bound: 398.8696133
time: 1.08 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8696133, upper bound: 398.8696133
time: 1.00 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 5.36 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.9007908, upper bound: 398.8825220
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.9118598, upper bound: 398.8927529
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.9007908, upper bound: 398.8825220
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.9118598, upper bound: 398.8927529
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8714421, upper bound: 398.8905354
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8909853, upper bound: 398.8909853
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8714421, upper bound: 398.8905354
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8909853, upper bound: 398.8909853
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8933698, upper bound: 398.8823622
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.9180232, upper bound: 398.8825575
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8933698, upper bound: 398.8823676
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.9180232, upper bound: 398.8825660
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8654150, upper bound: 398.8784810
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8895473, upper bound: 398.8787749
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8635442, upper bound: 398.8673649
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8891407, upper bound: 398.8675628
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8655450, upper bound: 398.8886404
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8822457, upper bound: 398.8898623
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8655450, upper bound: 398.8886404
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8822457, upper bound: 398.8898623
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8812696, upper bound: 398.8922058
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8708467, upper bound: 398.8916350
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8812696, upper bound: 398.8922058
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8708467, upper bound: 398.8916350
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8650748, upper bound: 398.8814831
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8820453, upper bound: 398.8819320
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8650748, upper bound: 398.8814831
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8820453, upper bound: 398.8819320
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8696133, upper bound: 398.8696133
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8696133, upper bound: 398.8696133
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8696133, upper bound: 398.8696133
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 0, lower bound: -398.8696133, upper bound: 398.8696133

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -143.9629669, 232.2097778, -152.7050781, 249.6213837, -393.5842896, 384.9148560
1: -158.2770081, 207.5072632, -167.9888458, 222.1666565, -380.4436340, 375.4960938
2: -157.9999237, 212.2220764, -167.7830505, 226.4978333, -384.4977417, 380.0051270
3: -186.7086945, 239.1380768, -198.3784180, 256.0103455, -442.7190552, 437.5164795
4: -160.7478485, 243.0447235, -170.7175293, 259.8674011, -420.6152344, 413.7622681

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9111631, upper bound: 398.8886113
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9112513, upper bound: 398.9103417
time: 1.29 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -168.5615845, 272.8850708, -157.6074982, 260.7996216, -429.3611450, 430.4925537
1: -185.7120361, 244.9842834, -173.5211639, 231.4784241, -417.1904602, 418.5054321
2: -185.3814392, 250.2069092, -173.3750916, 235.7609863, -421.1424255, 423.5820007
3: -219.6568146, 281.9783020, -205.2332764, 266.6873779, -486.3441772, 487.2115784
4: -189.2194977, 286.4551392, -176.6043701, 270.4530334, -459.6725464, 463.0595093

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9078663, upper bound: 398.8950825
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9127415, upper bound: 398.9127415
time: 1.19 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -143.9629669, 232.2097778, -175.0237579, 279.2850037, -423.2479858, 407.2335205
1: -158.2770081, 207.5072632, -191.9450378, 249.2747955, -407.5517883, 399.4522705
2: -157.9999237, 212.2220764, -192.0341644, 254.5219879, -412.5218201, 404.2562256
3: -186.7086945, 239.1380768, -225.5129547, 287.2160339, -473.9247437, 464.6510315
4: -160.7478485, 243.0447235, -194.2278748, 292.0315857, -452.7794189, 437.2725830

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8995212, upper bound: 398.8651335
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8991381, upper bound: 398.8758144
time: 1.13 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -168.5615845, 272.8850708, -178.4556580, 288.6043091, -457.1658325, 451.3407288
1: -185.7120361, 244.9842834, -195.8987885, 256.6459045, -442.3579407, 440.8830566
2: -185.3814392, 250.2069092, -196.0327301, 261.4757690, -446.8572083, 446.2396240
3: -219.6568146, 281.9783020, -230.5752563, 295.7846680, -515.4414673, 512.5534668
4: -189.2194977, 286.4551392, -198.4586182, 300.4693604, -489.6888428, 484.9137573

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9101293, upper bound: 398.8714295
time: 1.57 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9118490, upper bound: 398.8910630
time: 1.31 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -171.1241913, 273.7840576, -160.1047058, 264.5195618, -435.6437378, 433.8887329
1: -187.5600433, 243.4932709, -176.2011261, 234.6887360, -422.2487488, 419.6943970
2: -187.7182159, 247.8797760, -176.0927124, 238.9649048, -426.6831055, 423.9724731
3: -220.0472717, 280.7072754, -208.2464600, 270.3712463, -490.4184875, 488.9537354
4: -189.6266632, 285.1177368, -179.2045288, 274.2083130, -463.8349609, 464.3222351

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8651335, upper bound: 398.8995212
time: 1.33 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8714295, upper bound: 398.9101293
time: 1.45 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -181.0280762, 291.8794556, -160.8635864, 265.9017334, -446.9297791, 452.7430420
1: -198.6312103, 259.5796204, -177.0530396, 235.9219971, -434.5531311, 436.6326599
2: -198.8130798, 264.4493713, -176.9342804, 240.2380371, -439.0510864, 441.3836670
3: -233.6219635, 299.2090454, -209.2984161, 271.8085022, -505.4304810, 508.5074463
4: -201.1061249, 303.9932556, -180.0955811, 275.6540222, -476.7601318, 484.0888062

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8758144, upper bound: 398.8991381
time: 1.69 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8910630, upper bound: 398.9118490
time: 1.26 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -171.1241913, 273.7840576, -181.7697601, 293.2940674, -464.4182739, 455.5538025
1: -187.5600433, 243.4932709, -199.4552155, 260.8447876, -448.4048157, 442.9484253
2: -187.7182159, 247.8797760, -199.6382904, 265.6756592, -453.3938599, 447.5180054
3: -220.0472717, 280.7072754, -234.6184082, 300.6446838, -520.6919556, 515.3256836
4: -189.6266632, 285.1177368, -201.9862518, 305.4218445, -495.0484924, 487.1039734

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8713759, upper bound: 398.8713759
time: 1.05 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8713759, upper bound: 398.8905354
time: 1.14 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -181.0280762, 291.8794556, -182.5338745, 294.6768799, -475.7049561, 474.4133301
1: -198.6312103, 259.5796204, -200.3133240, 262.0391235, -460.6702271, 459.8929443
2: -198.8130798, 264.4493713, -200.4873199, 266.9307861, -465.7437744, 464.9367065
3: -233.6219635, 299.2090454, -235.6646729, 302.0430603, -535.6648560, 534.8737183
4: -201.1061249, 303.9932556, -202.8659515, 306.8235168, -507.9296265, 506.8591919

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8905089, upper bound: 398.8714421
time: 1.08 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8905089, upper bound: 398.8909853
time: 1.38 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -149.8447571, 244.7119904, -377.9041748, 569.1142578, -714.0614014, 608.8291016
1: -164.6911926, 217.2070312, -413.1052551, 520.7656250, -679.7973633, 618.1110840
2: -164.6050262, 220.8984070, -411.9991150, 529.9842529, -689.0591431, 622.8105469
3: -193.9470978, 250.2816925, -482.0190430, 601.0356445, -788.2874756, 722.7864380
4: -167.2314453, 253.8219147, -413.3740540, 607.0736084, -771.4949341, 658.2357788

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8881112, upper bound: 398.8797766
time: 1.42 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8931008, upper bound: 398.8654522
time: 2.84 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8931008, upper bound: 398.8823622
time: 1.32 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -159.3470764, 263.1299133, -379.0582581, 570.9978027, -725.6668701, 629.2648926
1: -175.3673859, 233.4939575, -414.4013672, 522.4734497, -692.5374756, 636.2485352
2: -175.2652588, 237.8358307, -413.2746582, 531.7001343, -701.5689697, 641.6648560
3: -207.2741852, 269.0414734, -483.5764160, 603.0059204, -803.9958496, 743.7389526
4: -178.3538513, 272.8920898, -414.6955566, 609.0549927, -785.0966187, 679.2182007

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9124156, upper bound: 398.8800182
time: 1.26 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9172709, upper bound: 398.8655893
time: 1.16 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9172709, upper bound: 398.8825575
time: 1.30 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -149.8447571, 244.7119904, -400.3330688, 601.0485229, -744.0825195, 628.9193726
1: -164.6911926, 217.2070312, -437.4002991, 550.4249268, -707.5648804, 640.2639771
2: -164.6050262, 220.8984070, -436.3471985, 560.3095093, -717.5457764, 645.0758057
3: -193.9470978, 250.2816925, -509.9006348, 635.4014893, -820.4324951, 749.0372925
4: -167.2314453, 253.8219147, -437.0654297, 641.6471558, -804.8280640, 680.3308105

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8887485, upper bound: 398.8799193
time: 1.35 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8849993, upper bound: 398.8675616
time: 1.24 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -159.3470764, 263.1299133, -403.0352478, 605.2098389, -757.8596191, 650.8104858
1: -175.3673859, 233.4939575, -440.3690186, 554.2597046, -722.3316650, 659.9888916
2: -175.2652588, 237.8358307, -439.3117371, 564.1909180, -732.1175537, 665.5360718
3: -207.2741852, 269.0414734, -513.3895874, 639.8392334, -838.4940796, 771.8417358
4: -178.3538513, 272.8920898, -440.0416870, 646.0687256, -820.7996216, 702.8915405

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9131712, upper bound: 398.8803440
time: 1.48 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9120030, upper bound: 398.8677172
time: 1.45 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -166.9178772, 267.3107910, -364.6539307, 548.1404419, -709.4222412, 617.1151123
1: -182.8971100, 237.4959106, -398.4037170, 501.0964966, -677.6203613, 623.3124390
2: -183.0760803, 241.7857666, -397.3427124, 510.1275024, -686.9270020, 628.5349731
3: -214.5748291, 273.7939148, -464.6038818, 578.5158081, -785.7850952, 728.6436157
4: -184.8644867, 278.1375427, -398.2894287, 584.2979736, -765.6981201, 667.1886597

Time for backsubstitution: 2.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8653583, upper bound: 398.8584923
time: 1.21 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8653583, upper bound: 398.8784810
time: 1.41 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -176.9484406, 285.6915894, -366.1846924, 550.7861938, -722.2174683, 638.1360474
1: -194.1059570, 253.8166656, -400.1183777, 503.4616699, -691.3885498, 642.1149902
2: -194.3089752, 258.6156921, -399.0569153, 512.5211182, -700.7197266, 647.9625854
3: -228.3164673, 292.5694275, -466.6952209, 581.2456055, -802.5487061, 750.4328003
4: -196.4939270, 297.2756042, -400.0623779, 587.0686646, -780.5076294, 688.6176758

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8895229, upper bound: 398.8587095
time: 1.60 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8895229, upper bound: 398.8787749
time: 1.15 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -169.7639618, 271.2597961, -374.6725159, 564.3008423, -728.9946899, 631.9206543
1: -186.0430908, 241.1886902, -409.6129150, 516.1083374, -696.3433838, 639.1082764
2: -186.2010956, 245.6366272, -408.4539490, 525.2591553, -705.7840576, 644.2880249
3: -218.2189941, 278.0820312, -477.9443665, 595.6713867, -807.1445312, 747.0466309
4: -188.0626678, 282.5010376, -409.8381042, 601.7373047, -786.8887939, 683.7882080

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=482.57733154296875
rel_dist={0: [-398.93352538884415, 398.93352538884415]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 1096.80 seconds
