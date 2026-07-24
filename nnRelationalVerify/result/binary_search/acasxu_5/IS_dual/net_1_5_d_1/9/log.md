## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_5.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 27.7691976323


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6.9769545, 23.4826069, -6.9769545, 23.4826069, -30.4595604, 30.4595604)
1: (-11.4056101, 24.0275745, -11.4056101, 24.0275745, -35.4331856, 35.4331856)
2: (-9.3722210, 25.8551235, -9.3722210, 25.8551235, -35.2273369, 35.2273369)
3: (-10.1196079, 35.4739799, -10.1196079, 35.4739799, -45.5935898, 45.5935898)
4: (-8.2898979, 33.6252289, -8.2898979, 33.6252289, -41.9151268, 41.9151268)

## BASE Result
execution time: IAR + LP analysis = 2.66 + 1.87 = 4.52 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -27.8527630, upper bound: 27.8527630


# Binary Search by BASE starts (time budget: 1195.48 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.2500000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=30.45956039428711
rel_dist={0: [-27.85276297632284, 27.852762976322843]}

## Binary search (step 1) starts
Candidate diff: 0.1250000


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=30.45956039428711
rel_dist={0: [-27.85276297632284, 27.852762976298393]}

## Binary search (step 2) starts
Candidate diff: 0.0625000


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=30.45956039428711
rel_dist={0: [-27.852403738376353, 27.852403738376353]}

## Binary search (step 3) starts
Candidate diff: 0.0312500


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=30.45956039428711
rel_dist={0: [-27.852018633109136, 27.85201863310914]}

## Binary search (step 4) starts
Candidate diff: 0.0156250


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=30.45956039428711
rel_dist={0: [-27.851803581362432, 27.851803581362432]}

## Binary search (step 5) starts
Candidate diff: 0.0078125


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0078125, mid=0.0078125, abs_max=30.45956039428711
rel_dist={0: [-27.851693595380517, 27.85169359538051]}

## Binary search (step 6) starts
Candidate diff: 0.0039062


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0039062, mid=0.0039062, abs_max=30.45956039428711
rel_dist={0: [-27.85163553907336, 27.851635539073364]}

## Binary search (step 7) starts
Candidate diff: 0.0019531


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0019531, mid=0.0019531, abs_max=30.45956039428711
rel_dist={0: [-27.85160631948645, 27.85160631948645]}

## Binary search (step 8) starts
Candidate diff: 0.0009766


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0009766, mid=0.0009766, abs_max=30.45956039428711
rel_dist={0: [-27.851590438334675, 27.851590438334668]}

## Binary search (step 9) starts
Candidate diff: 0.0004883


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0004883, mid=0.0004883, abs_max=30.45956039428711
rel_dist={0: [-27.85158212401153, 27.85158212401152]}

## Binary search (step 10) starts
Candidate diff: 0.0002441


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0002441, mid=0.0002441, abs_max=30.45956039428711
rel_dist={0: [-27.851577966852826, 27.85157796685283]}

## Binary search (step 11) starts
Candidate diff: 0.0001221


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0001221, mid=0.0001221, abs_max=30.45956039428711
rel_dist={0: [-27.8515758882792, 27.8515758882792]}

## Binary search (step 12) starts
Candidate diff: 0.0000610


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0000610, mid=0.0000610, abs_max=30.45956039428711
rel_dist={0: [-27.851574849003754, 27.85157484900374]}

## Binary search (step 13) starts
Candidate diff: 0.0000305


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000305, mid=0.0000305, abs_max=30.45956039428711
rel_dist={0: [-27.851574329388438, 27.851574329388427]}

## Binary search (step 14) starts
Candidate diff: 0.0000153


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000153, mid=0.0000153, abs_max=30.45956039428711
rel_dist={0: [-27.851574069624345, 27.851574069624334]}

## Binary search (step 15) starts
Candidate diff: 0.0000076


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000076, mid=0.0000076, abs_max=30.45956039428711
rel_dist={0: [-27.851573939824704, 27.851573939824696]}

## Binary search (step 16) starts
Candidate diff: 0.0000038


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000038, mid=0.0000038, abs_max=30.45956039428711
rel_dist={0: [-27.8515739029486, 27.851573875072916]}

## Binary search (step 17) starts
Candidate diff: 0.0000019


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000019, mid=0.0000019, abs_max=30.45956039428711
rel_dist={0: [-27.851573903270157, 27.851573924124807]}

## Binary search (step 18) starts
Candidate diff: 0.0000010


## IAR start
Binary search (step 18): status=Status.UNKNOWN, low=0.0000000, high=0.0000010, mid=0.0000010, abs_max=30.45956039428711
rel_dist={0: [-27.851573907432538, 27.851573896012034]}

## Binary Search Result
Binary search time: 86.31 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 1109.16 seconds

## Binary search (step 0) starts
Candidate diff: 0.2500000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8341428, upper bound: 27.8312140
time: 0.71 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8166164, upper bound: 27.8166164
time: 0.77 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.72 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.72
Output dim: 0, lower bound: -27.8341428, upper bound: 27.8312140
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.72
Output dim: 0, lower bound: -27.8166164, upper bound: 27.8166164

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -6.3772492, 21.5509911, -6.9769545, 23.4826069, -29.8598557, 28.5279408
1: -10.4523840, 21.9968719, -11.4056101, 24.0275745, -34.4799576, 33.4024811
2: -8.5577908, 23.7546215, -9.3722210, 25.8551235, -34.4129143, 33.1268349
3: -9.2616367, 32.6009521, -10.1196079, 35.4739799, -44.7356186, 42.7205582
4: -7.5485835, 30.8331852, -8.2898979, 33.6252289, -41.1738129, 39.1230850

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8327699, upper bound: 27.8302937
time: 0.89 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8182039, upper bound: 27.8180931
time: 0.71 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -6.8235641, 22.5012608, -6.8579302, 23.1025887, -29.9261532, 29.3591919
1: -11.1183262, 22.9801178, -11.2155113, 23.6299477, -34.7482681, 34.1956253
2: -9.1461391, 24.7790203, -9.2118444, 25.4422207, -34.5883522, 33.9908600
3: -9.8294525, 33.9572601, -9.9487162, 34.9049988, -44.7344475, 43.9059715
4: -8.0590858, 32.1531982, -8.1481714, 33.0771179, -41.1362038, 40.3013687

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8160573, upper bound: 27.8155351
time: 0.77 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8164844, upper bound: 27.8164844
time: 0.95 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.47 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 4.47
Output dim: 0, lower bound: -27.8327699, upper bound: 27.8302937
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 4.47
Output dim: 0, lower bound: -27.8182039, upper bound: 27.8180931
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 4.47
Output dim: 0, lower bound: -27.8160573, upper bound: 27.8155351
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 4.47
Output dim: 0, lower bound: -27.8164844, upper bound: 27.8164844

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -5.7200584, 19.5778465, -6.9769545, 23.4826069, -29.2026653, 26.5547981
1: -9.4149895, 19.9617805, -11.4056101, 24.0275745, -33.4425659, 31.3673897
2: -7.6789317, 21.5985489, -9.3722210, 25.8551235, -33.5340462, 30.9707680
3: -8.3492737, 29.6628551, -10.1196079, 35.4739799, -43.8232536, 39.7824631
4: -6.7730722, 27.9870510, -8.2898979, 33.6252289, -40.3983002, 36.2769470

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8172545, upper bound: 27.8176659
time: 0.87 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8172545, upper bound: 27.8180931
time: 0.64 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -8.2702923, 26.3485661, -6.9769545, 23.4826069, -31.7528992, 33.3255119
1: -13.3165970, 27.0048523, -11.4056101, 24.0275745, -37.3441696, 38.4104614
2: -11.0371552, 29.0161781, -9.3722210, 25.8551235, -36.8922691, 38.3883972
3: -11.7161083, 39.7636337, -10.1196079, 35.4739799, -47.1900826, 49.8832397
4: -9.7000866, 37.7317314, -8.2898979, 33.6252289, -43.3253174, 46.0216293

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8172545, upper bound: 27.8176659
time: 0.62 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8172545, upper bound: 27.8180931
time: 0.66 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -6.1421108, 20.3790970, -6.8579302, 23.1025887, -29.2446995, 27.2370262
1: -10.0364008, 20.8005562, -11.2155113, 23.6299477, -33.6663399, 32.0160637
2: -8.2307682, 22.4753113, -9.2118444, 25.4422207, -33.6729889, 31.6871529
3: -8.8714476, 30.8012981, -9.9487162, 34.9049988, -43.7764473, 40.7500153
4: -7.2492752, 29.1014862, -8.1481714, 33.0771179, -40.3263931, 37.2496567

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8160573, upper bound: 27.8155351
time: 0.98 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8160573, upper bound: 27.8155351
time: 0.70 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -8.1902189, 25.5940704, -6.8579302, 23.1025887, -31.2928085, 32.4519997
1: -13.1512871, 26.2427368, -11.2155113, 23.6299477, -36.7812233, 37.4582405
2: -10.9101553, 28.1654396, -9.2118444, 25.4422207, -36.3523750, 37.3772850
3: -11.5478954, 38.6183548, -9.9487162, 34.9049988, -46.4528885, 48.5670624
4: -9.5612869, 36.6561813, -8.1481714, 33.0771179, -42.6384048, 44.8043518

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8155351, upper bound: 27.8160573
time: 0.82 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8155351, upper bound: 27.8164844
time: 0.77 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.33 seconds
IS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 4.33
Output dim: 0, lower bound: -27.8172545, upper bound: 27.8176659
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 4.33
Output dim: 0, lower bound: -27.8172545, upper bound: 27.8180931
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 4.33
Output dim: 0, lower bound: -27.8172545, upper bound: 27.8176659
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 4.33
Output dim: 0, lower bound: -27.8172545, upper bound: 27.8180931
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 4.33
Output dim: 0, lower bound: -27.8160573, upper bound: 27.8155351
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 4.33
Output dim: 0, lower bound: -27.8160573, upper bound: 27.8155351
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 4.33
Output dim: 0, lower bound: -27.8155351, upper bound: 27.8160573
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 4.33
Output dim: 0, lower bound: -27.8155351, upper bound: 27.8164844

## BFS IS instance: IS_A1_A1_B1

### Backsubstitution after applying IS history:
0: -5.7200584, 19.5778465, -6.3009291, 21.4494267, -27.1694851, 25.8787746
1: -9.4149895, 19.9617805, -10.3412561, 21.9251156, -31.3401031, 30.3030357
2: -7.6789317, 21.5985489, -8.4684124, 23.6361237, -31.3150539, 30.0669613
3: -8.3492737, 29.6628551, -9.1817541, 32.4473190, -40.7965927, 38.8446083
4: -6.7730722, 27.9870510, -7.4937401, 30.6982441, -37.4713135, 35.4807892

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B1_A1

### Relational analysis result of IS_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8249468, upper bound: 27.8222676
time: 0.54 seconds

## Relational analysis of IS_A1_A1_B1_A2

### Relational analysis result of IS_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8232699, upper bound: 27.8211480
time: 0.61 seconds

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: -5.7200584, 19.5778465, -8.9866209, 28.6664867, -34.3865395, 28.5644646
1: -9.4149895, 19.9617805, -14.4501781, 29.4634552, -38.8784447, 34.4119568
2: -7.6789317, 21.5985489, -12.0248032, 31.5181274, -39.1970520, 33.6233521
3: -8.3492737, 29.6628551, -12.7294807, 43.1844177, -51.5336914, 42.3923340
4: -6.7730722, 27.9870510, -10.5634689, 40.9741936, -47.7472649, 38.5505180

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7970948, upper bound: 27.7846544
time: 0.93 seconds

## Relational analysis of IS_A1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B2_A1

### Relational analysis result of IS_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8249468, upper bound: 27.8222676
time: 0.57 seconds

## Relational analysis of IS_A1_A1_B2_A2

### Relational analysis result of IS_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8232699, upper bound: 27.8216564
time: 0.82 seconds

## BFS IS instance: IS_A1_A2_B1

### Backsubstitution after applying IS history:
0: -8.2702923, 26.3485661, -6.3009291, 21.4494267, -29.7197151, 32.6494865
1: -13.3165970, 27.0048523, -10.3412561, 21.9251156, -35.2417145, 37.3461075
2: -11.0371552, 29.0161781, -8.4684124, 23.6361237, -34.6732750, 37.4845886
3: -11.7161083, 39.7636337, -9.1817541, 32.4473190, -44.1634293, 48.9453812
4: -9.7000866, 37.7317314, -7.4937401, 30.6982441, -40.3983307, 45.2254715

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_A2_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.5466848, upper bound: 27.5804684
time: 0.65 seconds

## Relational analysis of IS_A1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_A2_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8172545, upper bound: 27.8176659
time: 0.68 seconds

## Relational analysis of IS_A1_A2_B1_B2

### Relational analysis result of IS_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8172545, upper bound: 27.8176659
time: 0.64 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: -8.2702923, 26.3485661, -8.9866209, 28.6664867, -36.9367790, 35.3351822
1: -13.3165970, 27.0048523, -14.4501781, 29.4634552, -42.7800522, 41.4550285
2: -11.0371552, 29.0161781, -12.0248032, 31.5181274, -42.5552826, 41.0409813
3: -11.7161083, 39.7636337, -12.7294807, 43.1844177, -54.9005203, 52.4931145
4: -9.7000866, 37.7317314, -10.5634689, 40.9741936, -50.6742783, 48.2952003

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.2156863, upper bound: 27.3717908
time: 0.99 seconds

## Relational analysis of IS_A1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_A2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8172545, upper bound: 27.8176659
time: 0.62 seconds

## Relational analysis of IS_A1_A2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8172545, upper bound: 27.8176659
time: 0.94 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: -6.1421108, 20.3790970, -6.3772492, 21.5509911, -27.6930962, 26.7563457
1: -10.0364008, 20.8005562, -10.4523840, 21.9968719, -32.0332718, 31.2529411
2: -8.2307682, 22.4753113, -8.5577908, 23.7546215, -31.9853897, 31.0331020
3: -8.8714476, 30.8012981, -9.2616367, 32.6009521, -41.4724007, 40.0629349
4: -7.2492752, 29.1014862, -7.5485835, 30.8331852, -38.0824509, 36.6500702

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8151079, upper bound: 27.8151079
time: 0.55 seconds

## Relational analysis of IS_A2_A1_B1_B2

### Relational analysis result of IS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8151079, upper bound: 27.8155351
time: 0.82 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: -6.1421108, 20.3790970, -6.8235641, 22.5012608, -28.6433697, 27.2026596
1: -10.0364008, 20.8005562, -11.1183262, 22.9801178, -33.0165138, 31.9188786
2: -8.2307682, 22.4753113, -9.1461391, 24.7790203, -33.0097885, 31.6214447
3: -8.8714476, 30.8012981, -9.8294525, 33.9572601, -42.8287086, 40.6307526
4: -7.2492752, 29.1014862, -8.0590858, 32.1531982, -39.4024658, 37.1605682

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A1_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8151079, upper bound: 27.8151079
time: 0.83 seconds

## Relational analysis of IS_A2_A1_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8151079, upper bound: 27.8155351
time: 0.75 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -8.1902189, 25.5940704, -6.1848683, 21.0785923, -29.2688103, 31.7789345
1: -13.1512871, 26.2427368, -10.1562214, 21.5374374, -34.6887169, 36.3989525
2: -10.9101553, 28.1654396, -8.3116837, 23.2333603, -34.1435165, 36.4771233
3: -11.5478954, 38.6183548, -9.0159798, 31.8923264, -43.4402237, 47.6343346
4: -9.5612869, 36.6561813, -7.3559322, 30.1641140, -39.7254028, 44.0121078

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8155351, upper bound: 27.8160573
time: 0.70 seconds

## Relational analysis of IS_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8155351, upper bound: 27.8160573
time: 0.91 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -8.1902189, 25.5940704, -8.8637781, 28.2822590, -36.4724731, 34.4578476
1: -13.1512871, 26.2427368, -14.2547560, 29.0595703, -42.2108459, 40.4974861
2: -10.9101553, 28.1654396, -11.8594084, 31.0981064, -42.0082626, 40.0248489
3: -11.5478954, 38.6183548, -12.5542936, 42.6112175, -54.1591072, 51.1726494
4: -9.5612869, 36.6561813, -10.4159803, 40.4193420, -49.9806252, 47.0721588

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8155351, upper bound: 27.8164844
time: 0.60 seconds

## Relational analysis of IS_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8155351, upper bound: 27.8164844
time: 0.85 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 6.70 seconds
IS_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 6.70
Output dim: 0, lower bound: -27.8249468, upper bound: 27.8222676
IS_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 6.70
Output dim: 0, lower bound: -27.8232699, upper bound: 27.8211480
IS_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 6.70
Output dim: 0, lower bound: -27.8249468, upper bound: 27.8222676
IS_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 6.70
Output dim: 0, lower bound: -27.8232699, upper bound: 27.8216564
IS_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 6.70
Output dim: 0, lower bound: -27.8172545, upper bound: 27.8176659
IS_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 6.70
Output dim: 0, lower bound: -27.8172545, upper bound: 27.8176659
IS_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 6.70
Output dim: 0, lower bound: -27.8172545, upper bound: 27.8176659
IS_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 6.70
Output dim: 0, lower bound: -27.8172545, upper bound: 27.8176659
IS_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 6.70
Output dim: 0, lower bound: -27.8151079, upper bound: 27.8151079
IS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 6.70
Output dim: 0, lower bound: -27.8151079, upper bound: 27.8155351
IS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 6.70
Output dim: 0, lower bound: -27.8151079, upper bound: 27.8151079
IS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 6.70
Output dim: 0, lower bound: -27.8151079, upper bound: 27.8155351
IS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 6.70
Output dim: 0, lower bound: -27.8155351, upper bound: 27.8160573
IS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 6.70
Output dim: 0, lower bound: -27.8155351, upper bound: 27.8160573
IS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 6.70
Output dim: 0, lower bound: -27.8155351, upper bound: 27.8164844
IS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 6.70
Output dim: 0, lower bound: -27.8155351, upper bound: 27.8164844

## BFS IS instance: IS_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4.4955974, 15.8341436, -6.2037916, 21.1608963, -25.6564941, 22.0379333
1: -7.4680882, 16.1045341, -10.1888885, 21.6262512, -29.0943394, 26.2934227
2: -6.0293436, 17.5095921, -8.3392849, 23.3215599, -29.3509026, 25.8488731
3: -6.6484203, 24.0976467, -9.0489531, 32.0225372, -38.6709557, 33.1465988
4: -5.3203859, 22.6204567, -7.3794670, 30.2877903, -35.6081734, 29.9999237

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B1_A1_A1

### Relational analysis result of IS_A1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8249478, upper bound: 27.8229510
time: 1.15 seconds

## Relational analysis of IS_A1_A1_B1_A1_A2

### Relational analysis result of IS_A1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8207059, upper bound: 27.8186598
time: 0.70 seconds

## BFS IS instance: IS_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -5.4029026, 18.6289616, -6.3009291, 21.4494267, -26.8523293, 24.9298897
1: -8.9004507, 18.9769077, -10.3412561, 21.9251156, -30.8255653, 29.3181648
2: -7.2595968, 20.5732021, -8.4684124, 23.6361237, -30.8957214, 29.0416107
3: -7.9015369, 28.2320576, -9.1817541, 32.4473190, -40.3488541, 37.4138031
4: -6.4061265, 26.6033630, -7.4937401, 30.6982441, -37.1043701, 34.0971031

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_A1_B1_A2_A1

### Relational analysis result of IS_A1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8117602, upper bound: 27.8115727
time: 0.83 seconds

## Relational analysis of IS_A1_A1_B1_A2_A2

### Relational analysis result of IS_A1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8232699, upper bound: 27.8211480
time: 0.58 seconds

## BFS IS instance: IS_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4.4955974, 15.8341436, -8.9203043, 28.4683800, -32.9639740, 24.7544441
1: -7.4680882, 16.1045341, -14.3458958, 29.2580357, -36.7261238, 30.4504242
2: -6.0293436, 17.5095921, -11.9363384, 31.3031807, -37.3325233, 29.4459305
3: -6.6484203, 24.0976467, -12.6382008, 42.8921394, -49.5405579, 36.7358475
4: -5.3203859, 22.6204567, -10.4868240, 40.6926346, -46.0130196, 33.1072807

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B2_A1_A1

### Relational analysis result of IS_A1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8250607, upper bound: 27.8222676
time: 0.84 seconds

## Relational analysis of IS_A1_A1_B2_A1_A2

### Relational analysis result of IS_A1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8219657, upper bound: 27.8191682
time: 0.67 seconds

## BFS IS instance: IS_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -5.4029026, 18.6289616, -8.9866209, 28.6664867, -34.0693893, 27.6155815
1: -8.9004507, 18.9769077, -14.4501781, 29.4634552, -38.3639069, 33.4270859
2: -7.2595968, 20.5732021, -12.0248032, 31.5181274, -38.7777214, 32.5980034
3: -7.9015369, 28.2320576, -12.7294807, 43.1844177, -51.0859528, 40.9615402
4: -6.4061265, 26.6033630, -10.5634689, 40.9741936, -47.3803215, 37.1668320

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_A1_B2_A2_A1

### Relational analysis result of IS_A1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8001971, upper bound: 27.7918365
time: 0.74 seconds

## Relational analysis of IS_A1_A1_B2_A2_A2

### Relational analysis result of IS_A1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8245296, upper bound: 27.8216564
time: 0.67 seconds

## BFS IS instance: IS_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -8.2702923, 26.3485661, -5.7200584, 19.5778465, -27.8481350, 32.0686150
1: -13.3165970, 27.0048523, -9.4149895, 19.9617805, -33.2783775, 36.4198418
2: -11.0371552, 29.0161781, -7.6789317, 21.5985489, -32.6357002, 36.6951103
3: -11.7161083, 39.7636337, -8.3492737, 29.6628551, -41.3789635, 48.1129036
4: -9.7000866, 37.7317314, -6.7730722, 27.9870510, -37.6871338, 44.5048027

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A2_B1_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8039856, upper bound: 27.8074719
time: 0.60 seconds

## Relational analysis of IS_A1_A2_B1_B1_B2

### Relational analysis result of IS_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8109195, upper bound: 27.8114979
time: 0.82 seconds

## BFS IS instance: IS_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -8.2702923, 26.3485661, -6.1218967, 20.3211250, -28.5914135, 32.4704590
1: -13.3165970, 27.0048523, -10.0048189, 20.7406521, -34.0572510, 37.0096703
2: -11.0371552, 29.0161781, -8.2034960, 22.4121971, -33.4493523, 37.2196693
3: -11.7161083, 39.7636337, -8.8431587, 30.7155533, -42.4316559, 48.6067924
4: -9.7000866, 37.7317314, -7.2250624, 29.0189571, -38.7190437, 44.9567947

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A2_B1_B2_B1

### Relational analysis result of IS_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8039856, upper bound: 27.8074719
time: 0.67 seconds

## Relational analysis of IS_A1_A2_B1_B2_B2

### Relational analysis result of IS_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8109195, upper bound: 27.8114979
time: 0.54 seconds

## BFS IS instance: IS_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -8.2702923, 26.3485661, -8.4906473, 27.0752850, -35.3455658, 34.8392105
1: -13.3165970, 27.0048523, -13.6665096, 27.7905121, -41.1071091, 40.6713562
2: -11.0371552, 29.0161781, -11.3479652, 29.7886066, -40.8257599, 40.3641434
3: -11.7161083, 39.7636337, -12.0308704, 40.8274689, -52.5435753, 51.7944984
4: -9.7000866, 37.7317314, -9.9613876, 38.7030602, -48.4031448, 47.6931190

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_B2_B1_B1

### Relational analysis result of IS_A1_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.2156863, upper bound: 27.3717908
time: 0.58 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B2_B1_B1

### Relational analysis result of IS_A1_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7517295, upper bound: 27.7383250
time: 0.90 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2

### Relational analysis result of IS_A1_A2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.6758105, upper bound: 27.6758105
time: 0.66 seconds

## BFS IS instance: IS_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -8.2702923, 26.3485661, -8.7929707, 27.6304836, -35.9007759, 35.1415329
1: -13.3165970, 27.0048523, -14.1271114, 28.3650951, -41.6816940, 41.1319656
2: -11.0371552, 29.0161781, -11.7445650, 30.3785744, -41.4157295, 40.7607422
3: -11.7161083, 39.7636337, -12.4195805, 41.6689415, -53.3850479, 52.1832085
4: -9.7000866, 37.7317314, -10.3061571, 39.5524216, -49.2525101, 48.0378876

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_B2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.2156863, upper bound: 27.3717908
time: 0.91 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B2_B2_A1

### Relational analysis result of IS_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7405219, upper bound: 27.7992826
time: 0.85 seconds

## Relational analysis of IS_A1_A2_B2_B2_A2

### Relational analysis result of IS_A1_A2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.6758105, upper bound: 27.7207494
time: 0.66 seconds

## BFS IS instance: IS_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -6.1421108, 20.3790970, -5.7200584, 19.5778465, -25.7199535, 26.0991554
1: -10.0364008, 20.8005562, -9.4149895, 19.9617805, -29.9981804, 30.2155437
2: -8.2307682, 22.4753113, -7.6789317, 21.5985489, -29.8293171, 30.1542435
3: -8.8714476, 30.8012981, -8.3492737, 29.6628551, -38.5343018, 39.1505737
4: -7.2492752, 29.1014862, -6.7730722, 27.9870510, -35.2363167, 35.8745537

Time for backsubstitution: 2.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_A1_B1_B1_A1

### Relational analysis result of IS_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8074719, upper bound: 27.8039856
time: 0.86 seconds

## Relational analysis of IS_A2_A1_B1_B1_A2

### Relational analysis result of IS_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8074719, upper bound: 27.8109195
time: 0.89 seconds

## BFS IS instance: IS_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -6.1421108, 20.3790970, -8.2702923, 26.3485661, -32.4906731, 28.6493855
1: -10.0364008, 20.8005562, -13.3165970, 27.0048523, -37.0412521, 34.1171532
2: -8.2307682, 22.4753113, -11.0371552, 29.0161781, -37.2469482, 33.5124626
3: -8.8714476, 30.8012981, -11.7161083, 39.7636337, -48.6350822, 42.5174065
4: -7.2492752, 29.1014862, -9.7000866, 37.7317314, -44.9810066, 38.8015747

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_A1_B1_B2_A1

### Relational analysis result of IS_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8074719, upper bound: 27.8039856
time: 1.30 seconds

## Relational analysis of IS_A2_A1_B1_B2_A2

### Relational analysis result of IS_A2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8114979, upper bound: 27.8109195
time: 0.61 seconds

## BFS IS instance: IS_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -6.1421108, 20.3790970, -6.1421108, 20.3790970, -26.5212059, 26.5212040
1: -10.0364008, 20.8005562, -10.0364008, 20.8005562, -30.8369522, 30.8369560
2: -8.2307682, 22.4753113, -8.2307682, 22.4753113, -30.7060795, 30.7060795
3: -8.8714476, 30.8012981, -8.8714476, 30.8012981, -39.6727448, 39.6727448
4: -7.2492752, 29.1014862, -7.2492752, 29.1014862, -36.3507538, 36.3507538

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_A1_B2_B1_A1

### Relational analysis result of IS_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8047094, upper bound: 27.8018015
time: 0.58 seconds

## Relational analysis of IS_A2_A1_B2_B1_A2

### Relational analysis result of IS_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8087354, upper bound: 27.8087354
time: 0.65 seconds

## BFS IS instance: IS_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -6.1421108, 20.3790970, -8.1902189, 25.5940704, -31.7361813, 28.5693169
1: -10.0364008, 20.8005562, -13.1512871, 26.2427368, -36.2791290, 33.9518356
2: -8.2307682, 22.4753113, -10.9101553, 28.1654396, -36.3962097, 33.3854675
3: -8.8714476, 30.8012981, -11.5478954, 38.6183548, -47.4898033, 42.3491936
4: -7.2492752, 29.1014862, -9.5612869, 36.6561813, -43.9054527, 38.6627693

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_A1_B2_B2_A1

### Relational analysis result of IS_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8047094, upper bound: 27.8023099
time: 0.91 seconds

## Relational analysis of IS_A2_A1_B2_B2_A2

### Relational analysis result of IS_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8087354, upper bound: 27.8092438
time: 1.37 seconds

## BFS IS instance: IS_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -8.1902189, 25.5940704, -5.7200584, 19.5778465, -27.7680664, 31.3141270
1: -13.1512871, 26.2427368, -9.4149895, 19.9617805, -33.1130600, 35.6577225
2: -10.9101553, 28.1654396, -7.6789317, 21.5985489, -32.5087051, 35.8443718
3: -11.5478954, 38.6183548, -8.3492737, 29.6628551, -41.2107468, 46.9676247
4: -9.5612869, 36.6561813, -6.7730722, 27.9870510, -37.5483284, 43.4292526

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_A2_B1_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8023099, upper bound: 27.8059692
time: 1.42 seconds

## Relational analysis of IS_A2_A2_B1_B1_B2

### Relational analysis result of IS_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8092438, upper bound: 27.8099951
time: 0.78 seconds

## BFS IS instance: IS_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -8.1902189, 25.5940704, -6.1218967, 20.3211250, -28.5113449, 31.7159634
1: -13.1512871, 26.2427368, -10.0048189, 20.7406521, -33.8919373, 36.2475548
2: -10.9101553, 28.1654396, -8.2034960, 22.4121971, -33.3223534, 36.3689308
3: -11.5478954, 38.6183548, -8.8431587, 30.7155533, -42.2634468, 47.4615135
4: -9.5612869, 36.6561813, -7.2250624, 29.0189571, -38.5802383, 43.8812370

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_A2_B1_B2_B1

### Relational analysis result of IS_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8023099, upper bound: 27.8059692
time: 0.76 seconds

## Relational analysis of IS_A2_A2_B1_B2_B2

### Relational analysis result of IS_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8092438, upper bound: 27.8099951
time: 1.04 seconds

## BFS IS instance: IS_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -8.1902189, 25.5940704, -8.4906473, 27.0752850, -35.2654953, 34.0847168
1: -13.1512871, 26.2427368, -13.6665096, 27.7905121, -40.9417953, 39.9092331
2: -10.9101553, 28.1654396, -11.3479652, 29.7886066, -40.6987610, 39.5134048
3: -11.5478954, 38.6183548, -12.0308704, 40.8274689, -52.3753586, 50.6492233
4: -9.5612869, 36.6561813, -9.9613876, 38.7030602, -48.2643471, 46.6175652

Time for backsubstitution: 2.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_A2_B2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7966685, upper bound: 27.7915260
time: 1.05 seconds

## Relational analysis of IS_A2_A2_B2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7207494, upper bound: 27.7290115
time: 0.75 seconds

## BFS IS instance: IS_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -8.1902189, 25.5940704, -8.7929707, 27.6304836, -35.8207016, 34.3870392
1: -13.1512871, 26.2427368, -14.1271114, 28.3650951, -41.5163803, 40.3698502
2: -10.9101553, 28.1654396, -11.7445650, 30.3785744, -41.2887306, 39.9100037
3: -11.5478954, 38.6183548, -12.4195805, 41.6689415, -53.2168312, 51.0379295
4: -9.5612869, 36.6561813, -10.3061571, 39.5524216, -49.1137085, 46.9623375

Time for backsubstitution: 2.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_A2_B2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7966685, upper bound: 27.7915260
time: 0.68 seconds

## Relational analysis of IS_A2_A2_B2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7207494, upper bound: 27.7739505
time: 0.89 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 11.76 seconds
IS_A1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 0, lower bound: -27.8249478, upper bound: 27.8229510
IS_A1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 0, lower bound: -27.8207059, upper bound: 27.8186598
IS_A1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 0, lower bound: -27.8117602, upper bound: 27.8115727
IS_A1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 0, lower bound: -27.8232699, upper bound: 27.8211480
IS_A1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 0, lower bound: -27.8250607, upper bound: 27.8222676
IS_A1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 0, lower bound: -27.8219657, upper bound: 27.8191682
IS_A1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 0, lower bound: -27.8001971, upper bound: 27.7918365
IS_A1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 0, lower bound: -27.8245296, upper bound: 27.8216564
IS_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 0, lower bound: -27.8039856, upper bound: 27.8074719
IS_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 0, lower bound: -27.8109195, upper bound: 27.8114979
IS_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 0, lower bound: -27.8039856, upper bound: 27.8074719
IS_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 0, lower bound: -27.8109195, upper bound: 27.8114979
IS_A1_A2_B2_B1_B1, status: Status.VERIFIED, split count: 5, time: 11.76
Output dim: 0, lower bound: -27.7517295, upper bound: 27.7383250
IS_A1_A2_B2_B1_B2, status: Status.VERIFIED, split count: 5, time: 11.76
Output dim: 0, lower bound: -27.6758105, upper bound: 27.6758105
IS_A1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 0, lower bound: -27.7405219, upper bound: 27.7992826
IS_A1_A2_B2_B2_A2, status: Status.VERIFIED, split count: 5, time: 11.76
Output dim: 0, lower bound: -27.6758105, upper bound: 27.7207494
IS_A2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 0, lower bound: -27.8074719, upper bound: 27.8039856
IS_A2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 0, lower bound: -27.8074719, upper bound: 27.8109195
IS_A2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 0, lower bound: -27.8074719, upper bound: 27.8039856
IS_A2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 0, lower bound: -27.8114979, upper bound: 27.8109195
IS_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 0, lower bound: -27.8047094, upper bound: 27.8018015
IS_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 0, lower bound: -27.8087354, upper bound: 27.8087354
IS_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 0, lower bound: -27.8047094, upper bound: 27.8023099
IS_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 0, lower bound: -27.8087354, upper bound: 27.8092438
IS_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 0, lower bound: -27.8023099, upper bound: 27.8059692
IS_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 0, lower bound: -27.8092438, upper bound: 27.8099951
IS_A2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 0, lower bound: -27.8023099, upper bound: 27.8059692
IS_A2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 0, lower bound: -27.8092438, upper bound: 27.8099951
IS_A2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 0, lower bound: -27.7966685, upper bound: 27.7915260
IS_A2_A2_B2_B1_B2, status: Status.VERIFIED, split count: 5, time: 11.76
Output dim: 0, lower bound: -27.7207494, upper bound: 27.7290115
IS_A2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 0, lower bound: -27.7966685, upper bound: 27.7915260
IS_A2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 11.76
Output dim: 0, lower bound: -27.7207494, upper bound: 27.7739505

## BFS IS instance: IS_A1_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -3.2067580, 12.1964846, -6.0810413, 20.8123894, -24.0191460, 18.2775249
1: -5.4015188, 12.2840137, -9.9961271, 21.2631130, -26.6646309, 22.2801361
2: -4.2838140, 13.4347467, -8.1758919, 22.9402637, -27.2240772, 21.6106377
3: -4.8055997, 18.5265827, -8.8795233, 31.5060482, -36.3116455, 27.4061050
4: -3.7714808, 17.1201935, -7.2340236, 29.7899437, -33.5614166, 24.3542175

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_A1_B1_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7927747, upper bound: 27.7895851
time: 0.80 seconds

## Relational analysis of IS_A1_A1_B1_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_A1_B1_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8249478, upper bound: 27.8229510
time: 0.89 seconds

## Relational analysis of IS_A1_A1_B1_A1_A1_B2

### Relational analysis result of IS_A1_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8249478, upper bound: 27.8229510
time: 0.73 seconds

## BFS IS instance: IS_A1_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -4.0146079, 14.3173351, -6.2037916, 21.1608963, -25.1755047, 20.5211258
1: -6.6892734, 14.5355740, -10.1888885, 21.6262512, -28.3155251, 24.7244625
2: -5.3807993, 15.8529587, -8.3392849, 23.3215599, -28.7023582, 24.1922417
3: -5.9516554, 21.8081436, -9.0489531, 32.0225372, -37.9741898, 30.8570976
4: -4.7432108, 20.4105949, -7.3794670, 30.2877903, -35.0310020, 27.7900620

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_A1_B1_A1_A2_B1

### Relational analysis result of IS_A1_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8207059, upper bound: 27.8186598
time: 0.69 seconds

## Relational analysis of IS_A1_A1_B1_A1_A2_B2

### Relational analysis result of IS_A1_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8207059, upper bound: 27.8186598
time: 0.56 seconds

## BFS IS instance: IS_A1_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -4.9169607, 17.0556355, -6.2417679, 21.2697411, -26.1866970, 23.2974014
1: -8.1146135, 17.3397942, -10.2471294, 21.7382450, -29.8528557, 27.5869217
2: -6.5825481, 18.8576317, -8.3882551, 23.4395199, -30.0220680, 27.2458858
3: -7.1810541, 25.8600273, -9.0980625, 32.1788368, -39.3598862, 34.9580879
4: -5.8216877, 24.3517532, -7.4224229, 30.4396114, -36.2612953, 31.7741718

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_A1_B1_A2_A1_B1

### Relational analysis result of IS_A1_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8117602, upper bound: 27.8115727
time: 0.61 seconds

## Relational analysis of IS_A1_A1_B1_A2_A1_B2

### Relational analysis result of IS_A1_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8117602, upper bound: 27.8115727
time: 0.89 seconds

## BFS IS instance: IS_A1_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -5.2254572, 18.0865288, -6.3009291, 21.4494267, -26.6748848, 24.3874569
1: -8.6175680, 18.4240227, -10.3412561, 21.9251156, -30.5426826, 28.7652779
2: -7.0218673, 19.9814758, -8.4684124, 23.6361237, -30.6579876, 28.4498863
3: -7.6518474, 27.4162827, -9.1817541, 32.4473190, -40.0991669, 36.5980301
4: -6.1995420, 25.8183708, -7.4937401, 30.6982441, -36.8977776, 33.3121109

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B1_A2_A2_A1

### Relational analysis result of IS_A1_A1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8232699, upper bound: 27.8211480
time: 0.71 seconds

## Relational analysis of IS_A1_A1_B1_A2_A2_A2

### Relational analysis result of IS_A1_A1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8221939, upper bound: 27.8210770
time: 0.82 seconds

## BFS IS instance: IS_A1_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -3.2067580, 12.1964846, -8.8443689, 28.2431736, -31.4499245, 21.0408478
1: -5.4015188, 12.2840137, -14.2261400, 29.0235214, -34.4250374, 26.5101490
2: -4.2838140, 13.4347467, -11.8348103, 31.0580063, -35.3418198, 25.2695560
3: -4.8055997, 18.5265827, -12.5324898, 42.5581818, -47.3637810, 31.0590725
4: -3.7714808, 17.1201935, -10.3982105, 40.3713188, -44.1427956, 27.5184040

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_A1_B2_A1_A1_B1

### Relational analysis result of IS_A1_A1_B2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7647242, upper bound: 27.7472965
time: 0.76 seconds

## Relational analysis of IS_A1_A1_B2_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_A1_B2_A1_A1_B1

### Relational analysis result of IS_A1_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8250607, upper bound: 27.8222676
time: 0.91 seconds

## Relational analysis of IS_A1_A1_B2_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8250607, upper bound: 27.8222676
time: 1.01 seconds

## BFS IS instance: IS_A1_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -4.0146079, 14.3173351, -8.9203043, 28.4683800, -32.4829865, 23.2376385
1: -6.6892734, 14.5355740, -14.3458958, 29.2580357, -35.9473076, 28.8814678
2: -5.3807993, 15.8529587, -11.9363384, 31.3031807, -36.6839790, 27.7892971
3: -5.9516554, 21.8081436, -12.6382008, 42.8921394, -48.8437920, 34.4463387
4: -4.7432108, 20.4105949, -10.4868240, 40.6926346, -45.4358444, 30.8974190

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_A1_B2_A1_A2_B1

### Relational analysis result of IS_A1_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8219657, upper bound: 27.8191682
time: 0.65 seconds

## Relational analysis of IS_A1_A1_B2_A1_A2_B2

### Relational analysis result of IS_A1_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8219657, upper bound: 27.8191682
time: 0.94 seconds

## BFS IS instance: IS_A1_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -4.9169607, 17.0556355, -8.9380684, 28.5182228, -33.4351845, 25.9937038
1: -8.1146135, 17.3397942, -14.3731880, 29.3092556, -37.4238701, 31.7129784
2: -6.5825481, 18.8576317, -11.9590893, 31.3562355, -37.9387817, 30.8167210
3: -7.1810541, 25.8600273, -12.6609583, 42.9635277, -50.1445770, 38.5209846
4: -5.8216877, 24.3517532, -10.5057831, 40.7615891, -46.5832710, 34.8575363

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_A1_B2_A2_A1_B1

### Relational analysis result of IS_A1_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8001971, upper bound: 27.7918365
time: 0.66 seconds

## Relational analysis of IS_A1_A1_B2_A2_A1_B2

### Relational analysis result of IS_A1_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8001971, upper bound: 27.7918365
time: 0.82 seconds

## BFS IS instance: IS_A1_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -5.2254572, 18.0865288, -8.9866209, 28.6664867, -33.8919373, 27.0731487
1: -8.6175680, 18.4240227, -14.4501781, 29.4634552, -38.0810204, 32.8741951
2: -7.0218673, 19.9814758, -12.0248032, 31.5181274, -38.5399933, 32.0062714
3: -7.6518474, 27.4162827, -12.7294807, 43.1844177, -50.8362656, 40.1457634
4: -6.1995420, 25.8183708, -10.5634689, 40.9741936, -47.1737289, 36.3818398

Time for backsubstitution: 2.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B2_A2_A2_A1

### Relational analysis result of IS_A1_A1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8245296, upper bound: 27.8216564
time: 0.73 seconds

## Relational analysis of IS_A1_A1_B2_A2_A2_A2

### Relational analysis result of IS_A1_A1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8234537, upper bound: 27.8215854
time: 0.80 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -8.2048597, 26.1545811, -4.4955974, 15.8341436, -24.0390015, 30.6501789
1: -13.2137661, 26.8042183, -7.4680882, 16.1045341, -29.3183002, 34.2723083
2: -10.9498119, 28.8050461, -6.0293436, 17.5095921, -28.4594021, 34.8343811
3: -11.6263866, 39.4771233, -6.6484203, 24.0976467, -35.7240334, 46.1255379
4: -9.6248074, 37.4555054, -5.3203859, 22.6204567, -32.2452621, 42.7758904

Time for backsubstitution: 2.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A2_B1_B1_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8243331, upper bound: 27.8275265
time: 0.72 seconds

## Relational analysis of IS_A1_A2_B1_B1_B1_B2

### Relational analysis result of IS_A1_A2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8208438, upper bound: 27.8234685
time: 1.12 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -8.2702923, 26.3485661, -5.4029026, 18.6289616, -26.8992538, 31.7514668
1: -13.3165970, 27.0048523, -8.9004507, 18.9769077, -32.2935028, 35.9053040
2: -11.0371552, 29.0161781, -7.2595968, 20.5732021, -31.6103573, 36.2757759
3: -11.7161083, 39.7636337, -7.9015369, 28.2320576, -39.9481621, 47.6651688
4: -9.7000866, 37.7317314, -6.4061265, 26.6033630, -36.3034515, 44.1378593

Time for backsubstitution: 2.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_A2_B1_B1_B2_B1

### Relational analysis result of IS_A1_A2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7949800, upper bound: 27.8061309
time: 0.92 seconds

## Relational analysis of IS_A1_A2_B1_B1_B2_B2

### Relational analysis result of IS_A1_A2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7949800, upper bound: 27.8260324
time: 0.69 seconds

## BFS IS instance: IS_A1_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -8.2048597, 26.1545811, -4.9192772, 16.7610512, -24.9659119, 31.0738564
1: -13.2137661, 26.8042183, -8.1127243, 17.0617619, -30.2755280, 34.9169426
2: -10.9498119, 28.8050461, -6.5915895, 18.5084305, -29.4582424, 35.3966293
3: -11.6263866, 39.4771233, -7.2023878, 25.4098148, -37.0361977, 46.6795082
4: -9.6248074, 37.4555054, -5.8034811, 23.9010277, -33.5258331, 43.2589874

Time for backsubstitution: 2.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_B1_A1

### Relational analysis result of IS_A1_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8020834, upper bound: 27.8047934
time: 0.83 seconds

## Relational analysis of IS_A1_A2_B1_B2_B1_A2

### Relational analysis result of IS_A1_A2_B1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7343328, upper bound: 27.7237753
time: 0.71 seconds

## BFS IS instance: IS_A1_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -8.2702923, 26.3485661, -5.7951403, 19.3845100, -27.6547985, 32.1437073
1: -13.3165970, 27.0048523, -9.4860306, 19.7728939, -33.0894928, 36.4908829
2: -11.0371552, 29.0161781, -7.7768764, 21.4003086, -32.4374619, 36.7930527
3: -11.7161083, 39.7636337, -8.3971539, 29.3123341, -41.0284424, 48.1607780
4: -9.7000866, 37.7317314, -6.8590794, 27.6703186, -37.3704071, 44.5908127

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_B1_B2_B2_B1

### Relational analysis result of IS_A1_A2_B1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.2073300, upper bound: 27.3647353
time: 0.65 seconds

## Relational analysis of IS_A1_A2_B1_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7925778, upper bound: 27.7976368
time: 0.99 seconds

## Relational analysis of IS_A1_A2_B1_B2_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7248273, upper bound: 27.7166187
time: 0.86 seconds

## BFS IS instance: IS_A1_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -8.1555071, 25.9988785, -8.7929707, 27.6304836, -35.7859879, 34.7918434
1: -13.1337318, 26.6443081, -14.1271114, 28.3650951, -41.4988251, 40.7714195
2: -10.8825235, 28.6351948, -11.7445650, 30.3785744, -41.2610970, 40.3797607
3: -11.5559359, 39.2406006, -12.4195805, 41.6689415, -53.2248764, 51.6601791
4: -9.5663986, 37.2266426, -10.3061571, 39.5524216, -49.1188202, 47.5327988

Time for backsubstitution: 2.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 45

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_B2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.1419599, upper bound: 27.3054017
time: 0.65 seconds

## Relational analysis of IS_A1_A2_B2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_A2_B2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.4317917, upper bound: 27.5074867
time: 0.61 seconds

## Relational analysis of IS_A1_A2_B2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7290115, upper bound: 27.7207494
time: 0.91 seconds

## Relational analysis of IS_A1_A2_B2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7290115, upper bound: 27.7207494
time: 0.90 seconds

## BFS IS instance: IS_A2_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -4.9192772, 16.7610512, -5.6237578, 19.2913780, -24.2106552, 22.3848095
1: -8.1127243, 17.0617619, -9.2637854, 19.6660786, -27.7788029, 26.3255463
2: -6.5915895, 18.5084305, -7.5508204, 21.2859612, -27.8775501, 26.0592499
3: -7.2023878, 25.4098148, -8.2175999, 29.2416592, -36.4440460, 33.6274147
4: -5.8034811, 23.9010277, -6.6601019, 27.5784035, -33.3818855, 30.5611305

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_A1_B1_B1_A1_B1

### Relational analysis result of IS_A2_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8171220, upper bound: 27.8163360
time: 0.90 seconds

## Relational analysis of IS_A2_A1_B1_B1_A1_B2

### Relational analysis result of IS_A2_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8170510, upper bound: 27.8152600
time: 0.68 seconds

## BFS IS instance: IS_A2_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -5.7951403, 19.3845100, -5.7200584, 19.5778465, -25.3729820, 25.1045685
1: -9.4860306, 19.7728939, -9.4149895, 19.9617805, -29.4478111, 29.1878834
2: -7.7768764, 21.4003086, -7.6789317, 21.5985489, -29.3754253, 29.0792389
3: -8.3971539, 29.3123341, -8.3492737, 29.6628551, -38.0600014, 37.6616058
4: -6.8590794, 27.6703186, -6.7730722, 27.9870510, -34.8461266, 34.4433861

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_A1_B1_B1_A2_B1

### Relational analysis result of IS_A2_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8211480, upper bound: 27.8232699
time: 0.67 seconds

## Relational analysis of IS_A2_A1_B1_B1_A2_B2

### Relational analysis result of IS_A2_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8210770, upper bound: 27.8221939
time: 0.79 seconds

## BFS IS instance: IS_A2_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -4.9192772, 16.7610512, -8.2048597, 26.1545811, -31.0738564, 24.9659119
1: -8.1127243, 17.0617619, -13.2137661, 26.8042183, -34.9169426, 30.2755280
2: -6.5915895, 18.5084305, -10.9498119, 28.8050461, -35.3966293, 29.4582424
3: -7.2023878, 25.4098148, -11.6263866, 39.4771233, -46.6795082, 37.0361938
4: -5.8034811, 23.9010277, -9.6248074, 37.4555054, -43.2589874, 33.5258331

Time for backsubstitution: 2.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_A1_B1_B2_A1_B1

### Relational analysis result of IS_A2_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8047934, upper bound: 27.8020834
time: 0.74 seconds

## Relational analysis of IS_A2_A1_B1_B2_A1_B2

### Relational analysis result of IS_A2_A1_B1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7237753, upper bound: 27.7343328
time: 0.72 seconds

## BFS IS instance: IS_A2_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -5.7951403, 19.3845100, -8.2702923, 26.3485661, -32.1437035, 27.6547985
1: -9.4860306, 19.7728939, -13.3165970, 27.0048523, -36.4908829, 33.0894928
2: -7.7768764, 21.4003086, -11.0371552, 29.0161781, -36.7930527, 32.4374619
3: -8.3971539, 29.3123341, -11.7161083, 39.7636337, -48.1607780, 41.0284424
4: -6.8590794, 27.6703186, -9.7000866, 37.7317314, -44.5908127, 37.3704033

Time for backsubstitution: 2.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_B1_B2_A2_A1

### Relational analysis result of IS_A2_A1_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.3647353, upper bound: 27.2073300
time: 0.72 seconds

## Relational analysis of IS_A2_A1_B1_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_A1_B1_B2_A2_B1

### Relational analysis result of IS_A2_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7976368, upper bound: 27.7925778
time: 0.85 seconds

## Relational analysis of IS_A2_A1_B1_B2_A2_B2

### Relational analysis result of IS_A2_A1_B1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7166187, upper bound: 27.7248273
time: 0.67 seconds

## BFS IS instance: IS_A2_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -4.9192772, 16.7610512, -6.0802073, 20.1947727, -25.1140499, 22.8412590
1: -8.1127243, 17.0617619, -9.9385843, 20.6084347, -28.7211590, 27.0003414
2: -6.5915895, 18.5084305, -8.1484165, 22.2749004, -28.8664894, 26.6568451
3: -7.2023878, 25.4098148, -8.7856445, 30.5290051, -37.7313881, 34.1954575
4: -5.8034811, 23.9010277, -7.1755905, 28.8384132, -34.6418953, 31.0766182

Time for backsubstitution: 2.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_A1_B2_B1_A1_B1

### Relational analysis result of IS_A2_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7977755, upper bound: 27.7977755
time: 0.76 seconds

## Relational analysis of IS_A2_A1_B2_B1_A1_B2

### Relational analysis result of IS_A2_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7977755, upper bound: 27.8018015
time: 1.03 seconds

## BFS IS instance: IS_A2_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -5.7951403, 19.3845100, -6.1421108, 20.3790970, -26.1742325, 25.5266190
1: -9.4860306, 19.7728939, -10.0364008, 20.8005562, -30.2865868, 29.8092957
2: -7.7768764, 21.4003086, -8.2307682, 22.4753113, -30.2521877, 29.6310749
3: -8.3971539, 29.3123341, -8.8714476, 30.8012981, -39.1984520, 38.1837807
4: -6.8590794, 27.6703186, -7.2492752, 29.1014862, -35.9605637, 34.9195862

Time for backsubstitution: 2.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_A1_B2_B1_A2_B1

### Relational analysis result of IS_A2_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8018015, upper bound: 27.8047094
time: 0.70 seconds

## Relational analysis of IS_A2_A1_B2_B1_A2_B2

### Relational analysis result of IS_A2_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8018015, upper bound: 27.8087354
time: 0.80 seconds

## BFS IS instance: IS_A2_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -4.9192772, 16.7610512, -8.1180801, 25.3783760, -30.2976532, 24.8791294
1: -8.1127243, 17.0617619, -13.0371408, 26.0190201, -34.1317444, 30.0988998
2: -6.5915895, 18.5084305, -10.8135700, 27.9307327, -34.5223122, 29.3219986
3: -7.2023878, 25.4098148, -11.4476471, 38.2987328, -45.5011139, 36.8574600
4: -5.8034811, 23.9010277, -9.4773483, 36.3473778, -42.1508598, 33.3783760

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_A1_B2_B2_A1_B1

### Relational analysis result of IS_A2_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8032784, upper bound: 27.8004078
time: 0.71 seconds

## Relational analysis of IS_A2_A1_B2_B2_A1_B2

### Relational analysis result of IS_A2_A1_B2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7237753, upper bound: 27.7343328
time: 0.97 seconds

## BFS IS instance: IS_A2_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -5.7951403, 19.3845100, -8.1902189, 25.5940704, -31.3892097, 27.5747299
1: -9.4860306, 19.7728939, -13.1512871, 26.2427368, -35.7287674, 32.9241753
2: -7.7768764, 21.4003086, -10.9101553, 28.1654396, -35.9423141, 32.3104630
3: -8.3971539, 29.3123341, -11.5478954, 38.6183548, -47.0155029, 40.8602295
4: -6.8590794, 27.6703186, -9.5612869, 36.6561813, -43.5152588, 37.2315979

Time for backsubstitution: 2.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_A1_B2_B2_A2_B1

### Relational analysis result of IS_A2_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7799873, upper bound: 27.7909022
time: 0.78 seconds

## Relational analysis of IS_A2_A1_B2_B2_A2_B2

### Relational analysis result of IS_A2_A1_B2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7166187, upper bound: 27.7635121
time: 0.82 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -8.1180801, 25.3783760, -4.4955974, 15.8341436, -23.9522171, 29.8739738
1: -13.0371408, 26.0190201, -7.4680882, 16.1045341, -29.1416702, 33.4871063
2: -10.8135700, 27.9307327, -6.0293436, 17.5095921, -28.3231564, 33.9600677
3: -11.4476471, 38.2987328, -6.6484203, 24.0976467, -35.5452957, 44.9471512
4: -9.4773483, 36.3473778, -5.3203859, 22.6204567, -32.0978050, 41.6677628

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_A2_B1_B1_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8222676, upper bound: 27.8250607
time: 0.76 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_B2

### Relational analysis result of IS_A2_A2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8191682, upper bound: 27.8219657
time: 0.64 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -8.1902189, 25.5940704, -5.4029026, 18.6289616, -26.8191795, 30.9969730
1: -13.1512871, 26.2427368, -8.9004507, 18.9769077, -32.1281891, 35.1431885
2: -10.9101553, 28.1654396, -7.2595968, 20.5732021, -31.4833565, 35.4250374
3: -11.5478954, 38.6183548, -7.9015369, 28.2320576, -39.7799454, 46.5198898
4: -9.5612869, 36.6561813, -6.4061265, 26.6033630, -36.1646461, 43.0623093

Time for backsubstitution: 2.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_A2_B1_B1_B2_A1

### Relational analysis result of IS_A2_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7599021, upper bound: 27.7720876
time: 1.08 seconds

## Relational analysis of IS_A2_A2_B1_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_A2_B1_B1_B2_A1

### Relational analysis result of IS_A2_A2_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.5812981, upper bound: 27.5500355
time: 0.77 seconds

## Relational analysis of IS_A2_A2_B1_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_A2_B1_B1_B2_A1

### Relational analysis result of IS_A2_A2_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.5362253, upper bound: 27.4964755
time: 0.67 seconds

## Relational analysis of IS_A2_A2_B1_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_A2_B1_B1_B2_B1

### Relational analysis result of IS_A2_A2_B1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7521088, upper bound: 27.7636721
time: 1.00 seconds

## Relational analysis of IS_A2_A2_B1_B1_B2_B2

### Relational analysis result of IS_A2_A2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8216564, upper bound: 27.8245296
time: 0.79 seconds

## BFS IS instance: IS_A2_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -8.1180801, 25.3783760, -4.9192772, 16.7610512, -24.8791294, 30.2976532
1: -13.0371408, 26.0190201, -8.1127243, 17.0617619, -30.0989037, 34.1317444
2: -10.8135700, 27.9307327, -6.5915895, 18.5084305, -29.3219986, 34.5223122
3: -11.4476471, 38.2987328, -7.2023878, 25.4098148, -36.8574600, 45.5011177
4: -9.4773483, 36.3473778, -5.8034811, 23.9010277, -33.3783760, 42.1508598

Time for backsubstitution: 2.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_A2_B1_B2_B1_A1

### Relational analysis result of IS_A2_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8004078, upper bound: 27.8032784
time: 0.79 seconds

## Relational analysis of IS_A2_A2_B1_B2_B1_A2

### Relational analysis result of IS_A2_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7799322, upper bound: 27.7775503
time: 0.65 seconds

## BFS IS instance: IS_A2_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -8.1902189, 25.5940704, -5.7951403, 19.3845100, -27.5747299, 31.3892097
1: -13.1512871, 26.2427368, -9.4860306, 19.7728939, -32.9241753, 35.7287674
2: -10.9101553, 28.1654396, -7.7768764, 21.4003086, -32.3104630, 35.9423141
3: -11.5478954, 38.6183548, -8.3971539, 29.3123341, -40.8602295, 47.0155029
4: -9.5612869, 36.6561813, -6.8590794, 27.6703186, -37.2315979, 43.5152588

Time for backsubstitution: 2.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_A2_B1_B2_B2_A1

### Relational analysis result of IS_A2_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7779798, upper bound: 27.7865527
time: 0.74 seconds

## Relational analysis of IS_A2_A2_B1_B2_B2_A2

### Relational analysis result of IS_A2_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7704266, upper bound: 27.7703937
time: 1.17 seconds

## BFS IS instance: IS_A2_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -8.1902189, 25.5940704, -8.3732862, 26.7181206, -34.9083405, 33.9673576
1: -13.1512871, 26.2427368, -13.4799461, 27.4211292, -40.5724106, 39.7226791
2: -10.9101553, 28.1654396, -11.1899624, 29.4001026, -40.3102570, 39.3554001
3: -11.5478954, 38.6183548, -11.8671751, 40.2939339, -51.8418236, 50.4855270
4: -9.5612869, 36.6561813, -9.8247614, 38.1887245, -47.7500076, 46.4809341

Time for backsubstitution: 2.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 45

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_A2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_A2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_A2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_A2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_A2_B2_B1_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.3054017, upper bound: 27.1419599
time: 0.72 seconds

## Relational analysis of IS_A2_A2_B2_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_A2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_A2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_A2_B2_B1_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.5074867, upper bound: 27.4317917
time: 0.80 seconds

## Relational analysis of IS_A2_A2_B2_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_A2_B2_B1_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7207494, upper bound: 27.7290115
time: 1.05 seconds

## Relational analysis of IS_A2_A2_B2_B1_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7207494, upper bound: 27.7290115
time: 1.27 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -8.1902189, 25.5940704, -8.6649303, 27.2381115, -35.4283295, 34.2590027
1: -13.1512871, 26.2427368, -13.9233932, 27.9603634, -41.1116486, 40.1661224
2: -10.9101553, 28.1654396, -11.5721560, 29.9520187, -40.8621750, 39.7375946
3: -11.5478954, 38.6183548, -12.2408266, 41.0825462, -52.6304359, 50.8591766
4: -9.5612869, 36.6561813, -10.1570692, 38.9870071, -48.5482941, 46.8132477

Time for backsubstitution: 2.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_A2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=30.45956039428711
rel_dist={0: [-27.85276297632284, 27.852762976322843]}

## Binary search (step 1) starts
Candidate diff: 0.1250000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8302092, upper bound: 27.8302446
time: 0.59 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8166164, upper bound: 27.8166164
time: 0.87 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.70 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 1.70
Output dim: 0, lower bound: -27.8302092, upper bound: 27.8302446
IS_B2, status: Status.UNKNOWN, split count: 1, time: 1.70
Output dim: 0, lower bound: -27.8166164, upper bound: 27.8166164

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -6.9769545, 23.4826069, -6.3772492, 21.5509911, -28.5279408, 29.8598518
1: -11.4056101, 24.0275745, -10.4523840, 21.9968719, -33.4024811, 34.4799576
2: -9.3722210, 25.8551235, -8.5577908, 23.7546215, -33.1268311, 34.4129143
3: -10.1196079, 35.4739799, -9.2616367, 32.6009521, -42.7205582, 44.7356186
4: -8.2898979, 33.6252289, -7.5485835, 30.8331852, -39.1230850, 41.1738129

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_B1_B1

### Relational analysis result of IS_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8271511, upper bound: 27.8272259
time: 0.63 seconds

## Relational analysis of IS_B1_B2

### Relational analysis result of IS_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8180931, upper bound: 27.8182039
time: 0.80 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -6.4522405, 21.8064003, -6.8235641, 22.5012608, -28.9534988, 28.6299648
1: -10.5669632, 22.2720280, -11.1183262, 22.9801178, -33.5470695, 33.3903542
2: -8.6663866, 24.0329170, -9.1461391, 24.7790203, -33.4454041, 33.1790504
3: -9.3674021, 32.9622688, -9.8294525, 33.9572601, -43.3246613, 42.7917175
4: -7.6638947, 31.2084999, -8.0590858, 32.1531982, -39.8170891, 39.2675743

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_B2_B1

### Relational analysis result of IS_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8155351, upper bound: 27.8160573
time: 0.88 seconds

## Relational analysis of IS_B2_B2

### Relational analysis result of IS_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8164844, upper bound: 27.8164844
time: 0.88 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.47 seconds
IS_B1_B1, status: Status.UNKNOWN, split count: 2, time: 4.47
Output dim: 0, lower bound: -27.8271511, upper bound: 27.8272259
IS_B1_B2, status: Status.UNKNOWN, split count: 2, time: 4.47
Output dim: 0, lower bound: -27.8180931, upper bound: 27.8182039
IS_B2_B1, status: Status.UNKNOWN, split count: 2, time: 4.47
Output dim: 0, lower bound: -27.8155351, upper bound: 27.8160573
IS_B2_B2, status: Status.UNKNOWN, split count: 2, time: 4.47
Output dim: 0, lower bound: -27.8164844, upper bound: 27.8164844

## BFS IS instance: IS_B1_B1

### Backsubstitution after applying IS history:
0: -6.8026190, 22.9597206, -5.7200584, 19.5778465, -26.3804646, 28.6797791
1: -11.1315269, 23.4866638, -9.4149895, 19.9617805, -31.0933075, 32.9016533
2: -9.1390448, 25.2844887, -7.6789317, 21.5985489, -30.7375946, 32.9634171
3: -9.8778009, 34.6955643, -8.3492737, 29.6628551, -39.5406532, 43.0448380
4: -8.0846577, 32.8726196, -6.7730722, 27.9870510, -36.0717049, 39.6456909

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B1_B1_A1

### Relational analysis result of IS_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8176659, upper bound: 27.8172545
time: 1.11 seconds

## Relational analysis of IS_B1_B1_A2

### Relational analysis result of IS_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8176659, upper bound: 27.8182039
time: 1.14 seconds

## BFS IS instance: IS_B1_B2

### Backsubstitution after applying IS history:
0: -6.5969415, 22.2671547, -8.2702923, 26.3485661, -32.9455070, 30.5374393
1: -10.7887554, 22.7641182, -13.3165970, 27.0048523, -37.7936058, 36.0807114
2: -8.8572235, 24.5377464, -11.0371552, 29.0161781, -37.8734016, 35.5749016
3: -9.5628567, 33.6382828, -11.7161083, 39.7636337, -49.3264923, 45.3543892
4: -7.8302336, 31.8534908, -9.7000866, 37.7317314, -45.5619659, 41.5535774

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B1_B2_A1

### Relational analysis result of IS_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8176659, upper bound: 27.8172545
time: 0.84 seconds

## Relational analysis of IS_B1_B2_A2

### Relational analysis result of IS_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8176659, upper bound: 27.8182039
time: 0.73 seconds

## BFS IS instance: IS_B2_B1

### Backsubstitution after applying IS history:
0: -6.2856283, 21.3050289, -6.1421108, 20.3790970, -26.6647224, 27.4471359
1: -10.3048801, 21.7555218, -10.0364008, 20.8005562, -31.1054306, 31.7919235
2: -8.4434013, 23.4862404, -8.2307682, 22.4753113, -30.9187069, 31.7170086
3: -9.1364088, 32.2173576, -8.8714476, 30.8012981, -39.9377060, 41.0888062
4: -7.4678736, 30.4866085, -7.2492752, 29.1014862, -36.5693512, 37.7358780

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_B1_A1

### Relational analysis result of IS_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8155351, upper bound: 27.8160573
time: 0.67 seconds

## Relational analysis of IS_B2_B1_A2

### Relational analysis result of IS_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8155351, upper bound: 27.8160573
time: 0.79 seconds

## BFS IS instance: IS_B2_B2

### Backsubstitution after applying IS history:
0: -6.0874119, 20.6526985, -8.1902189, 25.5940704, -31.6814823, 28.8429184
1: -9.9731693, 21.0755863, -13.1512871, 26.2427368, -36.2158966, 34.2268677
2: -8.1720448, 22.7764187, -10.9101553, 28.1654396, -36.3374863, 33.6865730
3: -8.8344650, 31.2178459, -11.5478954, 38.6183548, -47.4528122, 42.7657394
4: -7.2219343, 29.5219460, -9.5612869, 36.6561813, -43.8781128, 39.0832214

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B2_B2_A1

### Relational analysis result of IS_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8160573, upper bound: 27.8155351
time: 0.70 seconds

## Relational analysis of IS_B2_B2_A2

### Relational analysis result of IS_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8160573, upper bound: 27.8164844
time: 0.83 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.28 seconds
IS_B1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.28
Output dim: 0, lower bound: -27.8176659, upper bound: 27.8172545
IS_B1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.28
Output dim: 0, lower bound: -27.8176659, upper bound: 27.8182039
IS_B1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.28
Output dim: 0, lower bound: -27.8176659, upper bound: 27.8172545
IS_B1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.28
Output dim: 0, lower bound: -27.8176659, upper bound: 27.8182039
IS_B2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.28
Output dim: 0, lower bound: -27.8155351, upper bound: 27.8160573
IS_B2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.28
Output dim: 0, lower bound: -27.8155351, upper bound: 27.8160573
IS_B2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.28
Output dim: 0, lower bound: -27.8160573, upper bound: 27.8155351
IS_B2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.28
Output dim: 0, lower bound: -27.8160573, upper bound: 27.8164844

## BFS IS instance: IS_B1_B1_A1

### Backsubstitution after applying IS history:
0: -6.3009291, 21.4494267, -5.7200584, 19.5778465, -25.8787746, 27.1694851
1: -10.3412561, 21.9251156, -9.4149895, 19.9617805, -30.3030357, 31.3401051
2: -8.4684124, 23.6361237, -7.6789317, 21.5985489, -30.0669594, 31.3150482
3: -9.1817541, 32.4473190, -8.3492737, 29.6628551, -38.8446083, 40.7965927
4: -7.4937401, 30.6982441, -6.7730722, 27.9870510, -35.4807892, 37.4713135

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_B1_A1_B1

### Relational analysis result of IS_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8011850, upper bound: 27.8051689
time: 0.64 seconds

## Relational analysis of IS_B1_B1_A1_B2

### Relational analysis result of IS_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8194006, upper bound: 27.8194736
time: 0.68 seconds

## BFS IS instance: IS_B1_B1_A2

### Backsubstitution after applying IS history:
0: -8.9866209, 28.6664867, -5.7200584, 19.5778465, -28.5644646, 34.3865395
1: -14.4501781, 29.4634552, -9.4149895, 19.9617805, -34.4119568, 38.8784447
2: -12.0248032, 31.5181274, -7.6789317, 21.5985489, -33.6233482, 39.1970520
3: -12.7294807, 43.1844177, -8.3492737, 29.6628551, -42.3923340, 51.5336914
4: -10.5634689, 40.9741936, -6.7730722, 27.9870510, -38.5505142, 47.7472649

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_B1_A2_A1

### Relational analysis result of IS_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8271511, upper bound: 27.8272259
time: 0.87 seconds

## Relational analysis of IS_B1_B1_A2_A2

### Relational analysis result of IS_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8271511, upper bound: 27.8272259
time: 0.78 seconds

## BFS IS instance: IS_B1_B2_A1

### Backsubstitution after applying IS history:
0: -6.2992740, 21.4448643, -8.2702923, 26.3485661, -32.6478386, 29.7151489
1: -10.3385887, 21.9203682, -13.3165970, 27.0048523, -37.3434372, 35.2369652
2: -8.4662457, 23.6311302, -11.0371552, 29.0161781, -37.4824219, 34.6682854
3: -9.1793470, 32.4403687, -11.7161083, 39.7636337, -48.9429817, 44.1564674
4: -7.4917927, 30.6913986, -9.7000866, 37.7317314, -45.2235260, 40.3914871

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_B2_A1_A1

### Relational analysis result of IS_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8176659, upper bound: 27.8172545
time: 0.59 seconds

## Relational analysis of IS_B1_B2_A1_A2

### Relational analysis result of IS_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8176659, upper bound: 27.8172545
time: 0.67 seconds

## BFS IS instance: IS_B1_B2_A2

### Backsubstitution after applying IS history:
0: -8.9866209, 28.6664867, -8.2702923, 26.3485661, -35.3351822, 36.9367790
1: -14.4501781, 29.4634552, -13.3165970, 27.0048523, -41.4550323, 42.7800522
2: -12.0248032, 31.5181274, -11.0371552, 29.0161781, -41.0409813, 42.5552826
3: -12.7294807, 43.1844177, -11.7161083, 39.7636337, -52.4931145, 54.9005241
4: -10.5634689, 40.9741936, -9.7000866, 37.7317314, -48.2952003, 50.6742783

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_B2_A2_A1

### Relational analysis result of IS_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8176659, upper bound: 27.8182039
time: 0.63 seconds

## Relational analysis of IS_B1_B2_A2_A2

### Relational analysis result of IS_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8176659, upper bound: 27.8182039
time: 0.95 seconds

## BFS IS instance: IS_B2_B1_A1

### Backsubstitution after applying IS history:
0: -6.2042766, 21.0320816, -6.1421108, 20.3790970, -26.5833740, 27.1741886
1: -10.1794128, 21.4611855, -10.0364008, 20.8005562, -30.9799690, 31.4975853
2: -8.3263922, 23.1876450, -8.2307682, 22.4753113, -30.8017044, 31.4184132
3: -9.0215702, 31.8275528, -8.8714476, 30.8012981, -39.8228683, 40.6990013
4: -7.3444662, 30.0848503, -7.2492752, 29.1014862, -36.4459534, 37.3341179

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B2_B1_A1_A1

### Relational analysis result of IS_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8151079, upper bound: 27.8151079
time: 0.60 seconds

## Relational analysis of IS_B2_B1_A1_A2

### Relational analysis result of IS_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8151079, upper bound: 27.8160573
time: 0.59 seconds

## BFS IS instance: IS_B2_B1_A2

### Backsubstitution after applying IS history:
0: -6.6466398, 21.9462891, -6.1421108, 20.3790970, -27.0257378, 28.0883980
1: -10.8369436, 22.4105835, -10.0364008, 20.8005562, -31.6374912, 32.4469833
2: -8.9081955, 24.1761055, -8.2307682, 22.4753113, -31.3835068, 32.4068756
3: -9.5796223, 33.1310616, -8.8714476, 30.8012981, -40.3809204, 42.0025101
4: -7.8485918, 31.3541889, -7.2492752, 29.1014862, -36.9500771, 38.6034622

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 14

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B2_B1_A2_A1

### Relational analysis result of IS_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8151079, upper bound: 27.8151079
time: 0.83 seconds

## Relational analysis of IS_B2_B1_A2_A2

### Relational analysis result of IS_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8151079, upper bound: 27.8160573
time: 0.61 seconds

## BFS IS instance: IS_B2_B2_A1

### Backsubstitution after applying IS history:
0: -5.8023400, 19.8509502, -8.1902189, 25.5940704, -31.3964100, 28.0411682
1: -9.5436172, 20.2572060, -13.1512871, 26.2427368, -35.7863464, 33.4084892
2: -7.7976828, 21.8970871, -10.9101553, 28.1654396, -35.9631233, 32.8072433
3: -8.4672127, 30.0514793, -11.5478954, 38.6183548, -47.0855675, 41.5993729
4: -6.8992715, 28.3887100, -9.5612869, 36.6561813, -43.5554543, 37.9499893

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_B2_A1_A1

### Relational analysis result of IS_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8151079, upper bound: 27.8155351
time: 0.92 seconds

## Relational analysis of IS_B2_B2_A1_A2

### Relational analysis result of IS_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8151079, upper bound: 27.8155351
time: 0.73 seconds

## BFS IS instance: IS_B2_B2_A2

### Backsubstitution after applying IS history:
0: -7.7775364, 24.7857132, -8.1902189, 25.5940704, -33.3716049, 32.9759293
1: -12.5176897, 25.3961849, -13.1512871, 26.2427368, -38.7604218, 38.5474701
2: -10.3674421, 27.2983017, -10.9101553, 28.1654396, -38.5328827, 38.2084579
3: -11.0073185, 37.4021149, -11.5478954, 38.6183548, -49.6256714, 48.9500046
4: -9.1165113, 35.4650459, -9.5612869, 36.6561813, -45.7726936, 45.0263290

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_B2_A2_A1

### Relational analysis result of IS_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8151079, upper bound: 27.8164844
time: 0.95 seconds

## Relational analysis of IS_B2_B2_A2_A2

### Relational analysis result of IS_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8151079, upper bound: 27.8164844
time: 0.73 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 5.45 seconds
IS_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 0, lower bound: -27.8011850, upper bound: 27.8051689
IS_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 0, lower bound: -27.8194006, upper bound: 27.8194736
IS_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 0, lower bound: -27.8271511, upper bound: 27.8272259
IS_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 0, lower bound: -27.8271511, upper bound: 27.8272259
IS_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 0, lower bound: -27.8176659, upper bound: 27.8172545
IS_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 0, lower bound: -27.8176659, upper bound: 27.8172545
IS_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 0, lower bound: -27.8176659, upper bound: 27.8182039
IS_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 0, lower bound: -27.8176659, upper bound: 27.8182039
IS_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 0, lower bound: -27.8151079, upper bound: 27.8151079
IS_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 0, lower bound: -27.8151079, upper bound: 27.8160573
IS_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 0, lower bound: -27.8151079, upper bound: 27.8151079
IS_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 0, lower bound: -27.8151079, upper bound: 27.8160573
IS_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 0, lower bound: -27.8151079, upper bound: 27.8155351
IS_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 0, lower bound: -27.8151079, upper bound: 27.8155351
IS_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 0, lower bound: -27.8151079, upper bound: 27.8164844
IS_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 0, lower bound: -27.8151079, upper bound: 27.8164844

## BFS IS instance: IS_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -5.7662649, 19.8520069, -4.4955974, 15.8341436, -21.6004086, 24.3476048
1: -9.4995651, 20.2730732, -7.4680882, 16.1045341, -25.6040993, 27.7411613
2: -7.7577658, 21.8962288, -6.0293436, 17.5095921, -25.2673569, 27.9255695
3: -8.4471912, 30.0946083, -6.6484203, 24.0976467, -32.5448380, 36.7430267
4: -6.8657675, 28.4197521, -5.3203859, 22.6204567, -29.4862251, 33.7401352

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_B1_B1_A1_B1_B1

### Relational analysis result of IS_B1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8192101, upper bound: 27.8186033
time: 0.80 seconds

## Relational analysis of IS_B1_B1_A1_B1_B2

### Relational analysis result of IS_B1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8172824, upper bound: 27.8168658
time: 0.64 seconds

## BFS IS instance: IS_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6.3009291, 21.4494267, -5.4029026, 18.6289616, -24.9298897, 26.8523293
1: -10.3412561, 21.9251156, -8.9004507, 18.9769077, -29.3181629, 30.8255653
2: -8.4684124, 23.6361237, -7.2595968, 20.5732021, -29.0416145, 30.8957157
3: -9.1817541, 32.4473190, -7.9015369, 28.2320576, -37.4138031, 40.3488541
4: -7.4937401, 30.6982441, -6.4061265, 26.6033630, -34.0971031, 37.1043701

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_B1_A1_B2_A1

### Relational analysis result of IS_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8132341, upper bound: 27.8123551
time: 0.98 seconds

## Relational analysis of IS_B1_B1_A1_B2_A2

### Relational analysis result of IS_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8132341, upper bound: 27.8195647
time: 0.90 seconds

## BFS IS instance: IS_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -8.4906473, 27.0752850, -5.7200584, 19.5778465, -28.0684910, 32.7953300
1: -13.6665096, 27.7905121, -9.4149895, 19.9617805, -33.6282883, 37.2055016
2: -11.3479652, 29.7886066, -7.6789317, 21.5985489, -32.9465141, 37.4675331
3: -12.0308704, 40.8274689, -8.3492737, 29.6628551, -41.6937256, 49.1767426
4: -9.9613876, 38.7030602, -6.7730722, 27.9870510, -37.9484291, 45.4761314

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_B1_A2_A1_B1

### Relational analysis result of IS_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8011850, upper bound: 27.8051689
time: 1.08 seconds

## Relational analysis of IS_B1_B1_A2_A1_B2

### Relational analysis result of IS_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8194006, upper bound: 27.8194885
time: 1.04 seconds

## BFS IS instance: IS_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -8.7929707, 27.6304836, -5.7200584, 19.5778465, -28.3708172, 33.3505402
1: -14.1271114, 28.3650951, -9.4149895, 19.9617805, -34.0888901, 37.7800827
2: -11.7445650, 30.3785744, -7.6789317, 21.5985489, -33.3431129, 38.0575027
3: -12.4195805, 41.6689415, -8.3492737, 29.6628551, -42.0824356, 50.0182152
4: -10.3061571, 39.5524216, -6.7730722, 27.9870510, -38.2932053, 46.3254929

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_B1_A2_A2_B1

### Relational analysis result of IS_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8011850, upper bound: 27.8051689
time: 0.73 seconds

## Relational analysis of IS_B1_B1_A2_A2_B2

### Relational analysis result of IS_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8194006, upper bound: 27.8194885
time: 0.70 seconds

## BFS IS instance: IS_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -5.7200584, 19.5778465, -8.2702923, 26.3485661, -32.0686150, 27.8481350
1: -9.4149895, 19.9617805, -13.3165970, 27.0048523, -36.4198418, 33.2783775
2: -7.6789317, 21.5985489, -11.0371552, 29.0161781, -36.6951103, 32.6357002
3: -8.3492737, 29.6628551, -11.7161083, 39.7636337, -48.1129036, 41.3789635
4: -6.7730722, 27.9870510, -9.7000866, 37.7317314, -44.5048027, 37.6871338

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_B2_A1_A1_A1

### Relational analysis result of IS_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8011856, upper bound: 27.7984917
time: 0.81 seconds

## Relational analysis of IS_B1_B2_A1_A1_A2

### Relational analysis result of IS_B1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8114979, upper bound: 27.8109195
time: 0.83 seconds

## BFS IS instance: IS_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -6.1218967, 20.3211250, -8.2702923, 26.3485661, -32.4704590, 28.5914135
1: -10.0048189, 20.7406521, -13.3165970, 27.0048523, -37.0096703, 34.0572510
2: -8.2034960, 22.4121971, -11.0371552, 29.0161781, -37.2196693, 33.4493523
3: -8.8431587, 30.7155533, -11.7161083, 39.7636337, -48.6067924, 42.4316597
4: -7.2250624, 29.0189571, -9.7000866, 37.7317314, -44.9567947, 38.7190437

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_B2_A1_A2_A1

### Relational analysis result of IS_B1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8011856, upper bound: 27.7984917
time: 0.65 seconds

## Relational analysis of IS_B1_B2_A1_A2_A2

### Relational analysis result of IS_B1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8114979, upper bound: 27.8109195
time: 0.69 seconds

## BFS IS instance: IS_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -8.4906473, 27.0752850, -8.2702923, 26.3485661, -34.8392105, 35.3455620
1: -13.6665096, 27.7905121, -13.3165970, 27.0048523, -40.6713600, 41.1071091
2: -11.3479652, 29.7886066, -11.0371552, 29.0161781, -40.3641434, 40.8257599
3: -12.0308704, 40.8274689, -11.7161083, 39.7636337, -51.7944984, 52.5435753
4: -9.9613876, 38.7030602, -9.7000866, 37.7317314, -47.6931190, 48.4031448

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_B2_A2_A1_A1

### Relational analysis result of IS_B1_B2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7159605, upper bound: 27.7256740
time: 0.68 seconds

## Relational analysis of IS_B1_B2_A2_A1_A2

### Relational analysis result of IS_B1_B2_A2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.6758105, upper bound: 27.6758105
time: 0.78 seconds

## BFS IS instance: IS_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -8.7929707, 27.6304836, -8.2702923, 26.3485661, -35.1415329, 35.9007721
1: -14.1271114, 28.3650951, -13.3165970, 27.0048523, -41.1319656, 41.6816940
2: -11.7445650, 30.3785744, -11.0371552, 29.0161781, -40.7607422, 41.4157295
3: -12.4195805, 41.6689415, -11.7161083, 39.7636337, -52.1832085, 53.3850441
4: -10.3061571, 39.5524216, -9.7000866, 37.7317314, -48.0378876, 49.2525101

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_B2_A2_A2_B1

### Relational analysis result of IS_B1_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7256740, upper bound: 27.7653310
time: 0.62 seconds

## Relational analysis of IS_B1_B2_A2_A2_B2

### Relational analysis result of IS_B1_B2_A2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.6758105, upper bound: 27.7290115
time: 0.96 seconds

## BFS IS instance: IS_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -5.7160096, 19.5647163, -6.1421108, 20.3790970, -26.0951061, 25.7068233
1: -9.4085760, 19.9482117, -10.0364008, 20.8005562, -30.2091293, 29.9846115
2: -7.6733727, 21.5842686, -8.2307682, 22.4753113, -30.1486835, 29.8150368
3: -8.3435459, 29.6431007, -8.8714476, 30.8012981, -39.1448441, 38.5145493
4: -6.7682242, 27.9681129, -7.2492752, 29.1014862, -35.8697090, 35.2173882

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B2_B1_A1_A1_B1

### Relational analysis result of IS_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7984917, upper bound: 27.8011856
time: 0.78 seconds

## Relational analysis of IS_B2_B1_A1_A1_B2

### Relational analysis result of IS_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8109195, upper bound: 27.8114979
time: 0.85 seconds

## BFS IS instance: IS_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -7.8048763, 24.8208065, -6.1421108, 20.3790970, -28.1839733, 30.9629116
1: -12.5634775, 25.4264107, -10.0364008, 20.8005562, -33.3640327, 35.4628105
2: -10.3958282, 27.3481789, -8.2307682, 22.4753113, -32.8711395, 35.5789452
3: -11.0466223, 37.4607086, -8.8714476, 30.8012981, -41.8479195, 46.3321571
4: -9.1318951, 35.5199738, -7.2492752, 29.1014862, -38.2333794, 42.7692451

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B2_B1_A1_A2_B1

### Relational analysis result of IS_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7984917, upper bound: 27.8011856
time: 0.72 seconds

## Relational analysis of IS_B2_B1_A1_A2_B2

### Relational analysis result of IS_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8109195, upper bound: 27.8114979
time: 0.66 seconds

## BFS IS instance: IS_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -6.1421108, 20.3790970, -6.1421108, 20.3790970, -26.5212059, 26.5212040
1: -10.0364008, 20.8005562, -10.0364008, 20.8005562, -30.8369522, 30.8369560
2: -8.2307682, 22.4753113, -8.2307682, 22.4753113, -30.7060795, 30.7060795
3: -8.8714476, 30.8012981, -8.8714476, 30.8012981, -39.6727448, 39.6727448
4: -7.2492752, 29.1014862, -7.2492752, 29.1014862, -36.3507538, 36.3507538

Time for backsubstitution: 2.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B2_B1_A2_A1_B1

### Relational analysis result of IS_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8007890, upper bound: 27.8034425
time: 0.76 seconds

## Relational analysis of IS_B2_B1_A2_A1_B2

### Relational analysis result of IS_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8087354, upper bound: 27.8087354
time: 0.62 seconds

## BFS IS instance: IS_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -8.1902189, 25.5940704, -6.1421108, 20.3790970, -28.5693169, 31.7361774
1: -13.1512871, 26.2427368, -10.0364008, 20.8005562, -33.9518356, 36.2791290
2: -10.9101553, 28.1654396, -8.2307682, 22.4753113, -33.3854675, 36.3962097
3: -11.5478954, 38.6183548, -8.8714476, 30.8012981, -42.3491936, 47.4898033
4: -9.5612869, 36.6561813, -7.2492752, 29.1014862, -38.6627655, 43.9054527

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B2_B1_A2_A2_B1

### Relational analysis result of IS_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8007890, upper bound: 27.8034425
time: 0.70 seconds

## Relational analysis of IS_B2_B1_A2_A2_B2

### Relational analysis result of IS_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8087354, upper bound: 27.8099951
time: 0.61 seconds

## BFS IS instance: IS_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -5.7160096, 19.5647163, -8.1902189, 25.5940704, -31.3100796, 27.7549362
1: -9.4085760, 19.9482117, -13.1512871, 26.2427368, -35.6513100, 33.0994873
2: -7.6733727, 21.5842686, -10.9101553, 28.1654396, -35.8388138, 32.4944229
3: -8.3435459, 29.6431007, -11.5478954, 38.6183548, -46.9618988, 41.1909904
4: -6.7682242, 27.9681129, -9.5612869, 36.6561813, -43.4244041, 37.5293999

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_B2_A1_A1_A1

### Relational analysis result of IS_B2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8011856, upper bound: 27.8007890
time: 0.94 seconds

## Relational analysis of IS_B2_B2_A1_A1_A2

### Relational analysis result of IS_B2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8099951, upper bound: 27.8092438
time: 0.66 seconds

## BFS IS instance: IS_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -6.1218967, 20.3211250, -8.1902189, 25.5940704, -31.7159634, 28.5113449
1: -10.0048189, 20.7406521, -13.1512871, 26.2427368, -36.2475548, 33.8919373
2: -8.2034960, 22.4121971, -10.9101553, 28.1654396, -36.3689308, 33.3223534
3: -8.8431587, 30.7155533, -11.5478954, 38.6183548, -47.4615135, 42.2634468
4: -7.2250624, 29.0189571, -9.5612869, 36.6561813, -43.8812370, 38.5802383

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_B2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_B2_A1_A2_A1

### Relational analysis result of IS_B2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8011856, upper bound: 27.7985006
time: 0.57 seconds

## Relational analysis of IS_B2_B2_A1_A2_A2

### Relational analysis result of IS_B2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8099951, upper bound: 27.8092438
time: 0.72 seconds

## BFS IS instance: IS_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -7.8048763, 24.8208065, -8.1902189, 25.5940704, -33.3989449, 33.0110245
1: -12.5634775, 25.4264107, -13.1512871, 26.2427368, -38.8062134, 38.5776939
2: -10.3958282, 27.3481789, -10.9101553, 28.1654396, -38.5612679, 38.2583351
3: -11.0466223, 37.4607086, -11.5478954, 38.6183548, -49.6649780, 49.0086021
4: -9.1318951, 35.5199738, -9.5612869, 36.6561813, -45.7880745, 45.0812569

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_B2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_B2_A2_A1_A1

### Relational analysis result of IS_B2_B2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7159605, upper bound: 27.7657541
time: 0.78 seconds

## Relational analysis of IS_B2_B2_A2_A1_A2

### Relational analysis result of IS_B2_B2_A2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.6758105, upper bound: 27.7207494
time: 0.64 seconds

## BFS IS instance: IS_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -8.2580767, 25.8535194, -8.1902189, 25.5940704, -33.8521461, 34.0437355
1: -13.2682076, 26.5196285, -13.1512871, 26.2427368, -39.5109406, 39.6709061
2: -11.0042515, 28.4511890, -10.9101553, 28.1654396, -39.1696930, 39.3613434
3: -11.6544342, 39.0124893, -11.5478954, 38.6183548, -50.2727890, 50.5603790
4: -9.6563015, 37.0383072, -9.5612869, 36.6561813, -46.3124809, 46.5995941

Time for backsubstitution: 2.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_B2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_B2_A2_A2_B1

### Relational analysis result of IS_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7256740, upper bound: 27.7843897
time: 0.76 seconds

## Relational analysis of IS_B2_B2_A2_A2_B2

### Relational analysis result of IS_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.6758105, upper bound: 27.7739505
time: 0.60 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 11.66 seconds
IS_B1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 0, lower bound: -27.8192101, upper bound: 27.8186033
IS_B1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 0, lower bound: -27.8172824, upper bound: 27.8168658
IS_B1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 0, lower bound: -27.8132341, upper bound: 27.8123551
IS_B1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 0, lower bound: -27.8132341, upper bound: 27.8195647
IS_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 0, lower bound: -27.8011850, upper bound: 27.8051689
IS_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 0, lower bound: -27.8194006, upper bound: 27.8194885
IS_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 0, lower bound: -27.8011850, upper bound: 27.8051689
IS_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 0, lower bound: -27.8194006, upper bound: 27.8194885
IS_B1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 0, lower bound: -27.8011856, upper bound: 27.7984917
IS_B1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 0, lower bound: -27.8114979, upper bound: 27.8109195
IS_B1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 0, lower bound: -27.8011856, upper bound: 27.7984917
IS_B1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 0, lower bound: -27.8114979, upper bound: 27.8109195
IS_B1_B2_A2_A1_A1, status: Status.VERIFIED, split count: 5, time: 11.66
Output dim: 0, lower bound: -27.7159605, upper bound: 27.7256740
IS_B1_B2_A2_A1_A2, status: Status.VERIFIED, split count: 5, time: 11.66
Output dim: 0, lower bound: -27.6758105, upper bound: 27.6758105
IS_B1_B2_A2_A2_B1, status: Status.VERIFIED, split count: 5, time: 11.66
Output dim: 0, lower bound: -27.7256740, upper bound: 27.7653310
IS_B1_B2_A2_A2_B2, status: Status.VERIFIED, split count: 5, time: 11.66
Output dim: 0, lower bound: -27.6758105, upper bound: 27.7290115
IS_B2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 0, lower bound: -27.7984917, upper bound: 27.8011856
IS_B2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 0, lower bound: -27.8109195, upper bound: 27.8114979
IS_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 0, lower bound: -27.7984917, upper bound: 27.8011856
IS_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 0, lower bound: -27.8109195, upper bound: 27.8114979
IS_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 0, lower bound: -27.8007890, upper bound: 27.8034425
IS_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 0, lower bound: -27.8087354, upper bound: 27.8087354
IS_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 0, lower bound: -27.8007890, upper bound: 27.8034425
IS_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 0, lower bound: -27.8087354, upper bound: 27.8099951
IS_B2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 0, lower bound: -27.8011856, upper bound: 27.8007890
IS_B2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 0, lower bound: -27.8099951, upper bound: 27.8092438
IS_B2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 0, lower bound: -27.8011856, upper bound: 27.7985006
IS_B2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 0, lower bound: -27.8099951, upper bound: 27.8092438
IS_B2_B2_A2_A1_A1, status: Status.VERIFIED, split count: 5, time: 11.66
Output dim: 0, lower bound: -27.7159605, upper bound: 27.7657541
IS_B2_B2_A2_A1_A2, status: Status.VERIFIED, split count: 5, time: 11.66
Output dim: 0, lower bound: -27.6758105, upper bound: 27.7207494
IS_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 0, lower bound: -27.7256740, upper bound: 27.7843897
IS_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 0, lower bound: -27.6758105, upper bound: 27.7739505

## BFS IS instance: IS_B1_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -5.2344165, 18.2678699, -3.2067580, 12.1964846, -17.4309006, 21.4746284
1: -8.6499252, 18.6395607, -5.4015188, 12.2840137, -20.9339371, 24.0410805
2: -7.0455904, 20.1580563, -4.2838140, 13.4347467, -20.4803371, 24.4418697
3: -7.6946373, 27.7269001, -4.8055997, 18.5265827, -26.2212200, 32.5325012
4: -6.2289963, 26.1280956, -3.7714808, 17.1201935, -23.3491879, 29.8995762

Time for backsubstitution: 2.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_B1_A1_B1_B1_A1

### Relational analysis result of IS_B1_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8192101, upper bound: 27.8186033
time: 0.64 seconds

## Relational analysis of IS_B1_B1_A1_B1_B1_A2

### Relational analysis result of IS_B1_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8192101, upper bound: 27.8186033
time: 0.74 seconds

## BFS IS instance: IS_B1_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -5.7662649, 19.8520069, -4.0146079, 14.3173351, -20.0835991, 23.8666153
1: -9.4995651, 20.2730732, -6.6892734, 14.5355740, -24.0351391, 26.9623470
2: -7.7577658, 21.8962288, -5.3807993, 15.8529587, -23.6107254, 27.2770252
3: -8.4471912, 30.0946083, -5.9516554, 21.8081436, -30.2553349, 36.0462608
4: -6.8657675, 28.4197521, -4.7432108, 20.4105949, -27.2763634, 33.1629639

Time for backsubstitution: 2.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_B1_A1_B1_B2_A1

### Relational analysis result of IS_B1_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8172824, upper bound: 27.8168658
time: 0.62 seconds

## Relational analysis of IS_B1_B1_A1_B1_B2_A2

### Relational analysis result of IS_B1_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8172824, upper bound: 27.8168658
time: 0.76 seconds

## BFS IS instance: IS_B1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4.9898529, 17.4711838, -5.4029026, 18.6289616, -23.6188145, 22.8740864
1: -8.2712097, 17.8242245, -8.9004507, 18.9769077, -27.2481174, 26.7246723
2: -6.7083406, 19.2912483, -7.2595968, 20.5732021, -27.2815437, 26.5508442
3: -7.3757353, 26.5473442, -7.9015369, 28.2320576, -35.6077919, 34.4488831
4: -5.9454603, 25.0227203, -6.4061265, 26.6033630, -32.5488205, 31.4288445

Time for backsubstitution: 2.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_B1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_B1_B1_A1_B2_A1_B1

### Relational analysis result of IS_B1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8120118, upper bound: 27.8111771
time: 0.93 seconds

## Relational analysis of IS_B1_B1_A1_B2_A1_B2

### Relational analysis result of IS_B1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8132341, upper bound: 27.8123551
time: 0.97 seconds

## BFS IS instance: IS_B1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -5.9776387, 20.4788971, -5.4029026, 18.6289616, -24.6065998, 25.8817997
1: -9.8153677, 20.9132309, -8.9004507, 18.9769077, -28.7922745, 29.8136787
2: -8.0404854, 22.5888958, -7.2595968, 20.5732021, -28.6136875, 29.8484898
3: -8.7233858, 30.9885597, -7.9015369, 28.2320576, -36.9554405, 38.8900986
4: -7.1184216, 29.2894173, -6.4061265, 26.6033630, -33.7217865, 35.6955452

Time for backsubstitution: 2.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_B1_A1_B2_A2_A1

### Relational analysis result of IS_B1_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8132341, upper bound: 27.8195647
time: 0.79 seconds

## Relational analysis of IS_B1_B1_A1_B2_A2_A2

### Relational analysis result of IS_B1_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8132341, upper bound: 27.8195647
time: 0.71 seconds

## BFS IS instance: IS_B1_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -8.0920372, 25.8899899, -4.4955974, 15.8341436, -23.9261799, 30.3855877
1: -13.0397911, 26.5628490, -7.4680882, 16.1045341, -29.1443233, 34.0309334
2: -10.8157225, 28.5029850, -6.0293436, 17.5095921, -28.3253078, 34.5323181
3: -11.4836874, 39.0787392, -6.6484203, 24.0976467, -35.5813332, 45.7271576
4: -9.5027046, 37.0212479, -5.3203859, 22.6204567, -32.1231613, 42.3416328

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_B1_B1_A2_A1_B1_B1

### Relational analysis result of IS_B1_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7979621, upper bound: 27.8070032
time: 0.76 seconds

## Relational analysis of IS_B1_B1_A2_A1_B1_B2

### Relational analysis result of IS_B1_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7979621, upper bound: 27.8106569
time: 0.90 seconds

## BFS IS instance: IS_B1_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -8.4906473, 27.0752850, -5.4029026, 18.6289616, -27.1196079, 32.4781761
1: -13.6665096, 27.7905121, -8.9004507, 18.9769077, -32.6434135, 36.6909637
2: -11.3479652, 29.7886066, -7.2595968, 20.5732021, -31.9211674, 37.0481987
3: -12.0308704, 40.8274689, -7.9015369, 28.2320576, -40.2629242, 48.7290039
4: -9.9613876, 38.7030602, -6.4061265, 26.6033630, -36.5647469, 45.1091881

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B1_B1_A2_A1_B2_B1

### Relational analysis result of IS_B1_B1_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7361713, upper bound: 27.7473193
time: 0.63 seconds

## Relational analysis of IS_B1_B1_A2_A1_B2_B2

### Relational analysis result of IS_B1_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8213665, upper bound: 27.8220322
time: 0.80 seconds

## BFS IS instance: IS_B1_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -8.3778915, 26.3992443, -4.4955974, 15.8341436, -24.2120342, 30.8948421
1: -13.4741211, 27.0876961, -7.4680882, 16.1045341, -29.5786552, 34.5557785
2: -11.1894093, 29.0430775, -6.0293436, 17.5095921, -28.6989994, 35.0724144
3: -11.8482399, 39.8485260, -6.6484203, 24.0976467, -35.9458847, 46.4969482
4: -9.8261118, 37.7967987, -5.3203859, 22.6204567, -32.4465675, 43.1171837

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_B1_B1_A2_A2_B1_B1

### Relational analysis result of IS_B1_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7967841, upper bound: 27.8048002
time: 0.69 seconds

## Relational analysis of IS_B1_B1_A2_A2_B1_B2

### Relational analysis result of IS_B1_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8011850, upper bound: 27.8051689
time: 0.72 seconds

## BFS IS instance: IS_B1_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -8.7929707, 27.6304836, -5.4029026, 18.6289616, -27.4219322, 33.0333862
1: -14.1271114, 28.3650951, -8.9004507, 18.9769077, -33.1040192, 37.2655449
2: -11.7445650, 30.3785744, -7.2595968, 20.5732021, -32.3177681, 37.6381683
3: -12.4195805, 41.6689415, -7.9015369, 28.2320576, -40.6516342, 49.5704803
4: -10.3061571, 39.5524216, -6.4061265, 26.6033630, -36.9095192, 45.9585495

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_B1_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_B1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_B1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B1_B1_A2_A2_B2_A1

### Relational analysis result of IS_B1_B1_A2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.5459796, upper bound: 27.5066503
time: 0.67 seconds

## Relational analysis of IS_B1_B1_A2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B1_B1_A2_A2_B2_B1

### Relational analysis result of IS_B1_B1_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7022738, upper bound: 27.6982647
time: 1.93 seconds

## Relational analysis of IS_B1_B1_A2_A2_B2_B2

### Relational analysis result of IS_B1_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8194006, upper bound: 27.8194885
time: 0.96 seconds

## BFS IS instance: IS_B1_B2_A1_A1_A1

### Backsubstitution after applying IS history:
0: -4.4955974, 15.8341436, -7.8782840, 25.1840801, -29.6796780, 23.7124252
1: -7.4680882, 16.1045341, -12.7002869, 25.8013897, -33.2694778, 28.8048191
2: -6.0293436, 17.5095921, -10.5138378, 27.7520351, -33.7813759, 28.0234299
3: -6.6484203, 24.0976467, -11.1785927, 38.0448418, -44.6932602, 35.2762375
4: -5.3203859, 22.6204567, -9.2488480, 36.0759163, -41.3963013, 31.8693047

Time for backsubstitution: 2.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_B1_B2_A1_A1_A1_A1

### Relational analysis result of IS_B1_B2_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8070032, upper bound: 27.7979621
time: 0.79 seconds

## Relational analysis of IS_B1_B2_A1_A1_A1_A2

### Relational analysis result of IS_B1_B2_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8106569, upper bound: 27.8026398
time: 0.98 seconds

## BFS IS instance: IS_B1_B2_A1_A1_A2

### Backsubstitution after applying IS history:
0: -5.4029026, 18.6289616, -8.2702923, 26.3485661, -31.7514629, 26.8992538
1: -8.9004507, 18.9769077, -13.3165970, 27.0048523, -35.9053040, 32.2935028
2: -7.2595968, 20.5732021, -11.0371552, 29.0161781, -36.2757759, 31.6103497
3: -7.9015369, 28.2320576, -11.7161083, 39.7636337, -47.6651688, 39.9481621
4: -6.4061265, 26.6033630, -9.7000866, 37.7317314, -44.1378593, 36.3034515

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B1_B2_A1_A1_A2_A1

### Relational analysis result of IS_B1_B2_A1_A1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7473193, upper bound: 27.7361713
time: 0.77 seconds

## Relational analysis of IS_B1_B2_A1_A1_A2_A2

### Relational analysis result of IS_B1_B2_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8220322, upper bound: 27.8213665
time: 0.80 seconds

## BFS IS instance: IS_B1_B2_A1_A2_A1

### Backsubstitution after applying IS history:
0: -4.9192772, 16.7610512, -7.8782840, 25.1840801, -30.1033516, 24.6393356
1: -8.1127243, 17.0617619, -12.7002869, 25.8013897, -33.9141159, 29.7620487
2: -6.5915895, 18.5084305, -10.5138378, 27.7520351, -34.3436203, 29.0222683
3: -7.2023878, 25.4098148, -11.1785927, 38.0448418, -45.2472305, 36.5884056
4: -5.8034811, 23.9010277, -9.2488480, 36.0759163, -41.8793983, 33.1498756

Time for backsubstitution: 2.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_B1_B2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B1_B2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_B1_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_B2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B1_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B1_B2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_B2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_B1_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_B2_A1_A2_A1_B1

### Relational analysis result of IS_B1_B2_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7823959, upper bound: 27.7784164
time: 1.20 seconds

## Relational analysis of IS_B1_B2_A1_A2_A1_B2

### Relational analysis result of IS_B1_B2_A1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7224921, upper bound: 27.7326618
time: 0.64 seconds

## BFS IS instance: IS_B1_B2_A1_A2_A2

### Backsubstitution after applying IS history:
0: -5.7951403, 19.3845100, -8.2702923, 26.3485661, -32.1437035, 27.6547985
1: -9.4860306, 19.7728939, -13.3165970, 27.0048523, -36.4908829, 33.0894928
2: -7.7768764, 21.4003086, -11.0371552, 29.0161781, -36.7930527, 32.4374619
3: -8.3971539, 29.3123341, -11.7161083, 39.7636337, -48.1607780, 41.0284424
4: -6.8590794, 27.6703186, -9.7000866, 37.7317314, -44.5908127, 37.3704033

Time for backsubstitution: 2.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B1_B2_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_B1_B2_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_B1_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_B2_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B1_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B1_B2_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_B1_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B1_B2_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_B1_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_B1_B2_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_B2_A1_A2_A2_B1

### Relational analysis result of IS_B1_B2_A1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7688334, upper bound: 27.7678933
time: 0.65 seconds

## Relational analysis of IS_B1_B2_A1_A2_A2_B2

### Relational analysis result of IS_B1_B2_A1_A2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.6977736, upper bound: 27.7248273
time: 0.81 seconds

## BFS IS instance: IS_B2_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -5.2085848, 18.0425205, -4.9192772, 16.7610512, -21.9696350, 22.9617977
1: -8.6074123, 18.3792019, -8.1127243, 17.0617619, -25.6691742, 26.4919262
2: -6.9986181, 19.9259186, -6.5915895, 18.5084305, -25.5070457, 26.5175076
3: -7.6444225, 27.4002171, -7.2023878, 25.4098148, -33.0542297, 34.6026001
4: -6.1721554, 25.7896500, -5.8034811, 23.9010277, -30.0731831, 31.5931320

Time for backsubstitution: 2.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_B2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_B2_B1_A1_A1_B1_A1

### Relational analysis result of IS_B2_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8111771, upper bound: 27.8120118
time: 0.59 seconds

## Relational analysis of IS_B2_B1_A1_A1_B1_A2

### Relational analysis result of IS_B2_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8123551, upper bound: 27.8132341
time: 0.72 seconds

## BFS IS instance: IS_B2_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -5.7160096, 19.5647163, -5.7951403, 19.3845100, -25.1005192, 25.3598518
1: -9.4085760, 19.9482117, -9.4860306, 19.7728939, -29.1814690, 29.4342422
2: -7.6733727, 21.5842686, -7.7768764, 21.4003086, -29.0736809, 29.3611431
3: -8.3435459, 29.6431007, -8.3971539, 29.3123341, -37.6558800, 38.0402451
4: -6.7682242, 27.9681129, -6.8590794, 27.6703186, -34.4385376, 34.8271942

Time for backsubstitution: 2.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_B2_B1_A1_A1_B2_A1

### Relational analysis result of IS_B2_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8193587, upper bound: 27.8201109
time: 0.65 seconds

## Relational analysis of IS_B2_B1_A1_A1_B2_A2

### Relational analysis result of IS_B2_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8195646, upper bound: 27.8202651
time: 0.74 seconds

## BFS IS instance: IS_B2_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -7.4047551, 23.6257000, -4.9192772, 16.7610512, -24.1658058, 28.5449772
1: -11.9338627, 24.1915035, -8.1127243, 17.0617619, -28.9956226, 32.3042297
2: -9.8622265, 26.0489693, -6.5915895, 18.5084305, -28.3706570, 32.6405563
3: -10.4971075, 35.6935043, -7.2023878, 25.4098148, -35.9069214, 42.8958931
4: -8.6701927, 33.8190536, -5.8034811, 23.9010277, -32.5712204, 39.6225319

Time for backsubstitution: 2.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_B2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_B2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_B1_A1_A2_B1_A1

### Relational analysis result of IS_B2_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7784164, upper bound: 27.7823959
time: 0.84 seconds

## Relational analysis of IS_B2_B1_A1_A2_B1_A2

### Relational analysis result of IS_B2_B1_A1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7326618, upper bound: 27.7224921
time: 0.76 seconds

## BFS IS instance: IS_B2_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -7.8048763, 24.8208065, -5.7951403, 19.3845100, -27.1893864, 30.6159401
1: -12.5634775, 25.4264107, -9.4860306, 19.7728939, -32.3363724, 34.9124413
2: -10.3958282, 27.3481789, -7.7768764, 21.4003086, -31.7961369, 35.1250534
3: -11.0466223, 37.4607086, -8.3971539, 29.3123341, -40.3589554, 45.8578568
4: -9.1318951, 35.5199738, -6.8590794, 27.6703186, -36.8022118, 42.3790512

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_B2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_B2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_B2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_B2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_B1_A1_A2_B2_A1

### Relational analysis result of IS_B2_B1_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7678935, upper bound: 27.7688336
time: 0.69 seconds

## Relational analysis of IS_B2_B1_A1_A2_B2_A2

### Relational analysis result of IS_B2_B1_A1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7248273, upper bound: 27.7166187
time: 0.94 seconds

## BFS IS instance: IS_B2_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -5.7812662, 19.3051052, -4.9192772, 16.7610512, -22.5423164, 24.2243824
1: -9.4665422, 19.6817608, -8.1127243, 17.0617619, -26.5283012, 27.7944851
2: -7.7516751, 21.3075371, -6.5915895, 18.5084305, -26.2601051, 27.8991261
3: -8.3731089, 29.2157555, -7.2023878, 25.4098148, -33.7829170, 36.4181404
4: -6.8211994, 27.5709343, -5.8034811, 23.9010277, -30.7222271, 33.3744164

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_B2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_B2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_B1_A2_A1_B1_A1

### Relational analysis result of IS_B2_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7977755, upper bound: 27.7977755
time: 0.98 seconds

## Relational analysis of IS_B2_B1_A2_A1_B1_A2

### Relational analysis result of IS_B2_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7977755, upper bound: 27.8042448
time: 0.73 seconds

## BFS IS instance: IS_B2_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -6.1421108, 20.3790970, -5.7951403, 19.3845100, -25.5266190, 26.1742325
1: -10.0364008, 20.8005562, -9.4860306, 19.7728939, -29.8092957, 30.2865829
2: -8.2307682, 22.4753113, -7.7768764, 21.4003086, -29.6310768, 30.2521877
3: -8.8714476, 30.8012981, -8.3971539, 29.3123341, -38.1837807, 39.1984520
4: -7.2492752, 29.1014862, -6.8590794, 27.6703186, -34.9195862, 35.9605637

Time for backsubstitution: 2.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_B2_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_B2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_B1_A2_A1_B2_A1

### Relational analysis result of IS_B2_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8042448, upper bound: 27.8016529
time: 0.95 seconds

## Relational analysis of IS_B2_B1_A2_A1_B2_A2

### Relational analysis result of IS_B2_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8042448, upper bound: 27.8087354
time: 0.69 seconds

## BFS IS instance: IS_B2_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -7.7475805, 24.2730694, -4.9192772, 16.7610512, -24.5086327, 29.1923466
1: -12.4513187, 24.8740540, -8.1127243, 17.0617619, -29.5130768, 32.9867744
2: -10.3174686, 26.7297993, -6.5915895, 18.5084305, -28.8258972, 33.3213806
3: -10.9335709, 36.6614380, -7.2023878, 25.4098148, -36.3433838, 43.8638268
4: -9.0465889, 34.7658424, -5.8034811, 23.9010277, -32.9476166, 40.5693245

Time for backsubstitution: 2.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_B2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_B2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_B1_A2_A2_B1_A1

### Relational analysis result of IS_B2_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7852459, upper bound: 27.7915025
time: 0.77 seconds

## Relational analysis of IS_B2_B1_A2_A2_B1_A2

### Relational analysis result of IS_B2_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7781931, upper bound: 27.7762170
time: 0.87 seconds

## BFS IS instance: IS_B2_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -8.1902189, 25.5940704, -5.7951403, 19.3845100, -27.5747299, 31.3892097
1: -13.1512871, 26.2427368, -9.4860306, 19.7728939, -32.9241753, 35.7287674
2: -10.9101553, 28.1654396, -7.7768764, 21.4003086, -32.3104630, 35.9423141
3: -11.5478954, 38.6183548, -8.3971539, 29.3123341, -40.8602295, 47.0155029
4: -9.5612869, 36.6561813, -6.8590794, 27.6703186, -37.2315979, 43.5152588

Time for backsubstitution: 2.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_B2_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_B2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B2_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_B2_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_B2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_B2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_B1_A2_A2_B2_A1

### Relational analysis result of IS_B2_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7814257, upper bound: 27.7840325
time: 1.05 seconds

## Relational analysis of IS_B2_B1_A2_A2_B2_A2

### Relational analysis result of IS_B2_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7704266, upper bound: 27.7703937
time: 1.00 seconds

## BFS IS instance: IS_B2_B2_A1_A1_A1

### Backsubstitution after applying IS history:
0: -4.4955974, 15.8341436, -7.7475805, 24.2730694, -28.7686672, 23.5817242
1: -7.4680882, 16.1045341, -12.4513187, 24.8740540, -32.3421326, 28.5558491
2: -6.0293436, 17.5095921, -10.3174686, 26.7297993, -32.7591324, 27.8270607
3: -6.6484203, 24.0976467, -10.9335709, 36.6614380, -43.3098602, 35.0312195
4: -5.3203859, 22.6204567, -9.0465889, 34.7658424, -40.0862274, 31.6670456

Time for backsubstitution: 2.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_B2_B2_A1_A1_A1_A1

### Relational analysis result of IS_B2_B2_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8048002, upper bound: 27.7967841
time: 1.05 seconds

## Relational analysis of IS_B2_B2_A1_A1_A1_A2

### Relational analysis result of IS_B2_B2_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8048002, upper bound: 27.8011850
time: 1.41 seconds

## BFS IS instance: IS_B2_B2_A1_A1_A2

### Backsubstitution after applying IS history:
0: -5.3988667, 18.6157188, -8.1902189, 25.5940704, -30.9929371, 26.8059387
1: -8.8940010, 18.9632626, -13.1512871, 26.2427368, -35.1367340, 32.1145401
2: -7.2540541, 20.5587864, -10.9101553, 28.1654396, -35.4194946, 31.4689407
3: -7.8957767, 28.2121086, -11.5478954, 38.6183548, -46.5141296, 39.7599983
4: -6.4012489, 26.5842133, -9.5612869, 36.6561813, -43.0574303, 36.1455002

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B2_B2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_B2_B2_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_B2_B2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_B2_B2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B2_B2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_B2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_B2_B2_A1_A1_A2_B1

### Relational analysis result of IS_B2_B2_A1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.5066503, upper bound: 27.5459796
time: 0.66 seconds

## Relational analysis of IS_B2_B2_A1_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_B2_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_B2_A1_A1_A2_A1

### Relational analysis result of IS_B2_B2_A1_A1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.6982647, upper bound: 27.7022738
time: 1.10 seconds

## Relational analysis of IS_B2_B2_A1_A1_A2_A2

### Relational analysis result of IS_B2_B2_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8194885, upper bound: 27.8194006
time: 1.19 seconds

## BFS IS instance: IS_B2_B2_A1_A2_A1

### Backsubstitution after applying IS history:
0: -4.9192772, 16.7610512, -7.7475805, 24.2730694, -29.1923466, 24.5086327
1: -8.1127243, 17.0617619, -12.4513187, 24.8740540, -32.9867744, 29.5130730
2: -6.5915895, 18.5084305, -10.3174686, 26.7297993, -33.3213806, 28.8258991
3: -7.2023878, 25.4098148, -10.9335709, 36.6614380, -43.8638268, 36.3433838
4: -5.8034811, 23.9010277, -9.0465889, 34.7658424, -40.5693245, 32.9476166

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_B2_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_B2_B2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_B2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_B2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B2_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_B2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B2_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_B2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_B2_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_B2_A1_A2_A1_B1

### Relational analysis result of IS_B2_B2_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7905876, upper bound: 27.7888232
time: 0.79 seconds

## Relational analysis of IS_B2_B2_A1_A2_A1_B2

### Relational analysis result of IS_B2_B2_A1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7607951, upper bound: 27.7607880
time: 0.88 seconds

## BFS IS instance: IS_B2_B2_A1_A2_A2

### Backsubstitution after applying IS history:
0: -5.7951403, 19.3845100, -8.1902189, 25.5940704, -31.3892097, 27.5747299
1: -9.4860306, 19.7728939, -13.1512871, 26.2427368, -35.7287674, 32.9241753
2: -7.7768764, 21.4003086, -10.9101553, 28.1654396, -35.9423141, 32.3104630
3: -8.3971539, 29.3123341, -11.5478954, 38.6183548, -47.0155029, 40.8602295
4: -6.8590794, 27.6703186, -9.5612869, 36.6561813, -43.5152588, 37.2315979

Time for backsubstitution: 2.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_B2_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_B2_B2_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_B2_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_B2_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B2_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B2_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_B2_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_B2_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_B2_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_B2_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_B2_B2_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B2_B2_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_B2_A1_A2_A2_B1

### Relational analysis result of IS_B2_B2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7866355, upper bound: 27.7840598
time: 0.70 seconds

## Relational analysis of IS_B2_B2_A1_A2_A2_B2

### Relational analysis result of IS_B2_B2_A1_A2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7635119, upper bound: 27.7635119
time: 0.86 seconds

## BFS IS instance: IS_B2_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -8.2580767, 25.8535194, -8.0625410, 25.2043629, -33.4624290, 33.9160576
1: -13.2682076, 26.5196285, -12.9477577, 25.8417091, -39.1099167, 39.4673843
2: -11.0042515, 28.4511890, -10.7384300, 27.7407799, -38.7450294, 39.1896210
3: -11.6544342, 39.0124893, -11.3694210, 38.0350189, -49.6894531, 50.3819046
4: -9.6563015, 37.0383072, -9.4123125, 36.0927849, -45.7490845, 46.4506187

Time for backsubstitution: 2.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B2_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B2_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B2_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_B2_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_B2_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B2_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_B2_A2_A2_B1_A1

### Relational analysis result of IS_B2_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.79 seconds

## Relational analysis of IS_B2_B2_A2_A2_B1_A2

### Relational analysis result of IS_B2_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.81 seconds

## BFS IS instance: IS_B2_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -8.1860008, 25.6260452, -7.9677391, 24.8613853, -33.0473862, 33.5937767
1: -13.1529140, 26.2858486, -12.7934322, 25.4932976, -38.6462059, 39.0792809
2: -10.9073715, 28.2041836, -10.6108475, 27.3738270, -38.2811966, 38.8150330
3: -11.5530052, 38.6709251, -11.2339039, 37.5175400, -49.0705376, 49.9048309
4: -9.5718279, 36.7095833, -9.3000517, 35.6001701, -45.1719971, 46.0096321

Time for backsubstitution: 2.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B2_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B2_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_B2_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_B2_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B2_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B2_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_B2_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_B2_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B2_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B2_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_B2_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_B2_A2_A2_B2_A1

### Relational analysis result of IS_B2_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.96 seconds

## Relational analysis of IS_B2_B2_A2_A2_B2_A2

### Relational analysis result of IS_B2_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
time: 0.74 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 12.74 seconds
IS_B1_B1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.8192101, upper bound: 27.8186033
IS_B1_B1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.8192101, upper bound: 27.8186033
IS_B1_B1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.8172824, upper bound: 27.8168658
IS_B1_B1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.8172824, upper bound: 27.8168658
IS_B1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.8120118, upper bound: 27.8111771
IS_B1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.8132341, upper bound: 27.8123551
IS_B1_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.8132341, upper bound: 27.8195647
IS_B1_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.8132341, upper bound: 27.8195647
IS_B1_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.7979621, upper bound: 27.8070032
IS_B1_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.7979621, upper bound: 27.8106569
IS_B1_B1_A2_A1_B2_B1, status: Status.VERIFIED, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.7361713, upper bound: 27.7473193
IS_B1_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.8213665, upper bound: 27.8220322
IS_B1_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.7967841, upper bound: 27.8048002
IS_B1_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.8011850, upper bound: 27.8051689
IS_B1_B1_A2_A2_B2_B1, status: Status.VERIFIED, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.7022738, upper bound: 27.6982647
IS_B1_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.8194006, upper bound: 27.8194885
IS_B1_B2_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.8070032, upper bound: 27.7979621
IS_B1_B2_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.8106569, upper bound: 27.8026398
IS_B1_B2_A1_A1_A2_A1, status: Status.VERIFIED, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.7473193, upper bound: 27.7361713
IS_B1_B2_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.8220322, upper bound: 27.8213665
IS_B1_B2_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.7823959, upper bound: 27.7784164
IS_B1_B2_A1_A2_A1_B2, status: Status.VERIFIED, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.7224921, upper bound: 27.7326618
IS_B1_B2_A1_A2_A2_B1, status: Status.VERIFIED, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.7688334, upper bound: 27.7678933
IS_B1_B2_A1_A2_A2_B2, status: Status.VERIFIED, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.6977736, upper bound: 27.7248273
IS_B2_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.8111771, upper bound: 27.8120118
IS_B2_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.8123551, upper bound: 27.8132341
IS_B2_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.8193587, upper bound: 27.8201109
IS_B2_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.8195646, upper bound: 27.8202651
IS_B2_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.7784164, upper bound: 27.7823959
IS_B2_B1_A1_A2_B1_A2, status: Status.VERIFIED, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.7326618, upper bound: 27.7224921
IS_B2_B1_A1_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.7678935, upper bound: 27.7688336
IS_B2_B1_A1_A2_B2_A2, status: Status.VERIFIED, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.7248273, upper bound: 27.7166187
IS_B2_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.7977755, upper bound: 27.7977755
IS_B2_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.7977755, upper bound: 27.8042448
IS_B2_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.8042448, upper bound: 27.8016529
IS_B2_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.8042448, upper bound: 27.8087354
IS_B2_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.7852459, upper bound: 27.7915025
IS_B2_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.7781931, upper bound: 27.7762170
IS_B2_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.7814257, upper bound: 27.7840325
IS_B2_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.7704266, upper bound: 27.7703937
IS_B2_B2_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.8048002, upper bound: 27.7967841
IS_B2_B2_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.8048002, upper bound: 27.8011850
IS_B2_B2_A1_A1_A2_A1, status: Status.VERIFIED, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.6982647, upper bound: 27.7022738
IS_B2_B2_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.8194885, upper bound: 27.8194006
IS_B2_B2_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.7905876, upper bound: 27.7888232
IS_B2_B2_A1_A2_A1_B2, status: Status.VERIFIED, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.7607951, upper bound: 27.7607880
IS_B2_B2_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.7866355, upper bound: 27.7840598
IS_B2_B2_A1_A2_A2_B2, status: Status.VERIFIED, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.7635119, upper bound: 27.7635119
IS_B2_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
IS_B2_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
IS_B2_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505
IS_B2_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 12.74
Output dim: 0, lower bound: -27.7739505, upper bound: 27.7739505

## BFS IS instance: IS_B1_B1_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -4.7047215, 16.5330753, -3.2067580, 12.1964846, -16.9012051, 19.7398319
1: -7.7999063, 16.8236294, -5.4015188, 12.2840137, -20.0839195, 22.2251472
2: -6.3250041, 18.2756557, -4.2838140, 13.4347467, -19.7597504, 22.5594692
3: -6.9297419, 25.1477966, -4.8055997, 18.5265827, -25.4563217, 29.9533958
4: -5.5705442, 23.6083565, -3.7714808, 17.1201935, -22.6907349, 27.3798370

Time for backsubstitution: 2.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_B1_B1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_B1_B1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B1_B1_A1_B1_B1_A1_A1

### Relational analysis result of IS_B1_B1_A1_B1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7478570, upper bound: 27.7557968
time: 0.64 seconds

## Relational analysis of IS_B1_B1_A1_B1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B1_B1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B1_B1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_B1_B1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_B1_A1_B1_B1_A1_A1

### Relational analysis result of IS_B1_B1_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8152125, upper bound: 27.8149728
time: 0.71 seconds

## Relational analysis of IS_B1_B1_A1_B1_B1_A1_A2

### Relational analysis result of IS_B1_B1_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8152125, upper bound: 27.8186033
time: 0.61 seconds

## BFS IS instance: IS_B1_B1_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -5.0058851, 17.2783031, -3.2067580, 12.1964846, -17.2023640, 20.4850597
1: -8.2614260, 17.5635014, -5.4015188, 12.2840137, -20.5454311, 22.9650192
2: -6.7227206, 19.0781364, -4.2838140, 13.4347467, -20.1574669, 23.3619499
3: -7.3244295, 26.2049904, -4.8055997, 18.5265827, -25.8510094, 31.0105858
4: -5.9080873, 24.6577015, -3.7714808, 17.1201935, -23.0282764, 28.4291821

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_B1_B1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_B1_B1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B1_B1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_B1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B1_B1_A1_B1_B1_A2_A1

### Relational analysis result of IS_B1_B1_A1_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7478570, upper bound: 27.7557968
time: 0.60 seconds

## Relational analysis of IS_B1_B1_A1_B1_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_B1_B1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_B1_A1_B1_B1_A2_A1

### Relational analysis result of IS_B1_B1_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8152125, upper bound: 27.8149728
time: 0.56 seconds

## Relational analysis of IS_B1_B1_A1_B1_B1_A2_A2

### Relational analysis result of IS_B1_B1_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8152125, upper bound: 27.8186033
time: 0.59 seconds

## BFS IS instance: IS_B1_B1_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -5.2126913, 18.0557861, -4.0146079, 14.3173351, -19.5300255, 22.0703945
1: -8.6138868, 18.3928776, -6.6892734, 14.5355740, -23.1494579, 25.0821514
2: -7.0042686, 19.9404144, -5.3807993, 15.8529587, -22.8572254, 25.3212109
3: -7.6502028, 27.4201164, -5.9516554, 21.8081436, -29.4583435, 33.3717690
4: -6.1770940, 25.8087559, -4.7432108, 20.4105949, -26.5876884, 30.5519543

Time for backsubstitution: 2.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_B1_B1_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_B1_B1_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B1_B1_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B1_B1_A1_B1_B2_A1_A1

### Relational analysis result of IS_B1_B1_A1_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7460490, upper bound: 27.7540875
time: 0.66 seconds

## Relational analysis of IS_B1_B1_A1_B1_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B1_B1_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_B1_B1_A1_B1_B2_A1_A1

### Relational analysis result of IS_B1_B1_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8138664, upper bound: 27.8131553
time: 0.91 seconds

## Relational analysis of IS_B1_B1_A1_B1_B2_A1_A2

### Relational analysis result of IS_B1_B1_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8138664, upper bound: 27.8168658
time: 0.86 seconds

## BFS IS instance: IS_B1_B1_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -5.7740774, 19.2868080, -4.0146079, 14.3173351, -20.0914116, 23.3014164
1: -9.4555950, 19.6601410, -6.6892734, 14.5355740, -23.9911671, 26.3494148
2: -7.7427988, 21.2885742, -5.3807993, 15.8529587, -23.5957565, 26.6693726
3: -8.3634529, 29.1912498, -5.9516554, 21.8081436, -30.1715965, 35.1429062
4: -6.8137627, 27.5473843, -4.7432108, 20.4105949, -27.2243576, 32.2905960

Time for backsubstitution: 2.58 seconds
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=30.45956039428711
rel_dist={0: [-27.85276297632284, 27.852762976298393]}

## Binary search (step 2) starts
Candidate diff: 0.0625000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8397763, upper bound: 27.8388867
time: 0.75 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8373771, upper bound: 27.8373771
time: 0.72 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.70 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.70
Output dim: 0, lower bound: -27.8397763, upper bound: 27.8388867
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.70
Output dim: 0, lower bound: -27.8373771, upper bound: 27.8373771

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -5.4061508, 18.7729778, -5.9441805, 20.4128990, -25.8190460, 24.7171574
1: -8.9430733, 19.1648998, -9.7853575, 20.8546658, -29.7977390, 28.9502563
2: -7.2751994, 20.7168579, -8.0000391, 22.5083332, -29.7835331, 28.7168941
3: -7.9728169, 28.5126152, -8.7043095, 30.9467201, -38.9195366, 37.2169266
4: -6.4483242, 26.9256763, -7.0782318, 29.2423496, -35.6906738, 34.0039024

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7845550, upper bound: 27.7798004
time: 1.20 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -26.6935482, upper bound: 26.6665646
time: 0.67 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -6.6130686, 22.3832588, -6.8179593, 23.0023842, -29.6154499, 29.2012177
1: -10.8148479, 22.8833485, -11.1480122, 23.5276985, -34.3425446, 34.0313606
2: -8.8901510, 24.6683903, -9.1608162, 25.3366547, -34.2268066, 33.8292007
3: -9.6036949, 33.8211365, -9.8940830, 34.7529068, -44.3565979, 43.7152176
4: -7.8671532, 32.0281372, -8.1050024, 32.9279175, -40.7950706, 40.1331367

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8373040, upper bound: 27.8373771
time: 0.95 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8373040, upper bound: 27.8373771
time: 0.87 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.56 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 4.56
Output dim: 0, lower bound: -27.7845550, upper bound: 27.7798004
IS_A1_A2, status: Status.VERIFIED, split count: 2, time: 4.56
Output dim: 0, lower bound: -26.6935482, upper bound: 26.6665646
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.56
Output dim: 0, lower bound: -27.8373040, upper bound: 27.8373771
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.56
Output dim: 0, lower bound: -27.8373040, upper bound: 27.8373771

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -4.9898529, 17.4711838, -5.6797023, 19.6034641, -24.5933170, 23.1508846
1: -8.2712097, 17.8242245, -9.3638115, 20.0214329, -28.2926426, 27.1880341
2: -6.7083406, 19.2912483, -7.6439552, 21.6237774, -28.3321190, 26.9352036
3: -7.3757353, 26.5473442, -8.3323269, 29.7337723, -37.1095047, 34.8796692
4: -5.9454603, 25.0227203, -6.7630382, 28.0668793, -34.0123405, 31.7857590

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7654962, upper bound: 27.7640621
time: 0.80 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7845550, upper bound: 27.7798004
time: 0.71 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -6.6130686, 22.3832588, -5.4061508, 18.7729778, -25.3860474, 27.7894058
1: -10.8148479, 22.8833485, -8.9430733, 19.1648998, -29.9797401, 31.8264217
2: -8.8901510, 24.6683903, -7.2751994, 20.7168579, -29.6070080, 31.9435902
3: -9.6036949, 33.8211365, -7.9728169, 28.5126152, -38.1163101, 41.7939529
4: -7.8671532, 32.0281372, -6.4483242, 26.9256763, -34.7928314, 38.4764633

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7798004, upper bound: 27.7845549
time: 0.69 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -26.6665646, upper bound: 26.6935481
time: 0.80 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -6.6130686, 22.3832588, -6.6130686, 22.3832588, -28.9963264, 28.9963264
1: -10.8148479, 22.8833485, -10.8148479, 22.8833485, -33.6981926, 33.6981926
2: -8.8901510, 24.6683903, -8.8901510, 24.6683903, -33.5585403, 33.5585403
3: -9.6036949, 33.8211365, -9.6036949, 33.8211365, -43.4248314, 43.4248314
4: -7.8671532, 32.0281372, -7.8671532, 32.0281372, -39.8952904, 39.8952904

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.4166686, upper bound: 27.4658075
time: 0.88 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.3227097, upper bound: 27.3227097
time: 0.61 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.28 seconds
IS_A1_A1_B1, status: Status.VERIFIED, split count: 3, time: 4.28
Output dim: 0, lower bound: -27.7654962, upper bound: 27.7640621
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 4.28
Output dim: 0, lower bound: -27.7845550, upper bound: 27.7798004
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 4.28
Output dim: 0, lower bound: -27.7798004, upper bound: 27.7845549
IS_A2_B1_B2, status: Status.VERIFIED, split count: 3, time: 4.28
Output dim: 0, lower bound: -26.6665646, upper bound: 26.6935481
IS_A2_B2_B1, status: Status.VERIFIED, split count: 3, time: 4.28
Output dim: 0, lower bound: -27.4166686, upper bound: 27.4658075
IS_A2_B2_B2, status: Status.VERIFIED, split count: 3, time: 4.28
Output dim: 0, lower bound: -27.3227097, upper bound: 27.3227097

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: -4.2426796, 15.0208941, -5.6578388, 18.9135571, -23.1562347, 20.6787281
1: -7.0561638, 15.2701111, -9.2717447, 19.2887726, -26.3449364, 24.5418549
2: -5.6916437, 16.6246243, -7.5918212, 20.8814888, -26.5731316, 24.2164459
3: -6.2818270, 22.8508835, -8.2086658, 28.6347141, -34.9165382, 31.0595474
4: -5.0319147, 21.4414806, -6.6798763, 27.0115948, -32.0435066, 28.1213570

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_A1_B2_A1

### Relational analysis result of IS_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7767951, upper bound: 27.7748354
time: 0.75 seconds

## Relational analysis of IS_A1_A1_B2_A2

### Relational analysis result of IS_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7767951, upper bound: 27.7798004
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -6.2677369, 21.3492565, -4.9898529, 17.4711838, -23.7389202, 26.3391094
1: -10.2714300, 21.8131447, -8.2712097, 17.8242245, -28.0956535, 30.0843544
2: -8.4283066, 23.5389862, -6.7083406, 19.2912483, -27.7195549, 30.2473259
3: -9.1244440, 32.2827950, -7.3757353, 26.5473442, -35.6717758, 39.6585312
4: -7.4601769, 30.5399647, -5.9454603, 25.0227203, -32.4828987, 36.4854240

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7640621, upper bound: 27.7654962
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7798004, upper bound: 27.7845549
time: 1.13 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 5.47 seconds
IS_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 5.47
Output dim: 0, lower bound: -27.7767951, upper bound: 27.7748354
IS_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 5.47
Output dim: 0, lower bound: -27.7767951, upper bound: 27.7798004
IS_A2_B1_B1_A1, status: Status.VERIFIED, split count: 4, time: 5.47
Output dim: 0, lower bound: -27.7640621, upper bound: 27.7654962
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 5.47
Output dim: 0, lower bound: -27.7798004, upper bound: 27.7845549

## BFS IS instance: IS_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4.4955974, 15.8341436, -5.6578388, 18.9135571, -23.4091549, 21.4919758
1: -7.4680882, 16.1045341, -9.2717447, 19.2887726, -26.7568607, 25.3762779
2: -6.0293436, 17.5095921, -7.5918212, 20.8814888, -26.9108315, 25.1014099
3: -6.6484203, 24.0976467, -8.2086658, 28.6347141, -35.2831306, 32.3063126
4: -5.3203859, 22.6204567, -6.6798763, 27.0115948, -32.3319817, 29.3003330

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B2_A1_B1

### Relational analysis result of IS_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7553153, upper bound: 27.7748354
time: 1.14 seconds

## Relational analysis of IS_A1_A1_B2_A1_B2

### Relational analysis result of IS_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7553153, upper bound: 27.7748354
time: 0.84 seconds

## BFS IS instance: IS_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -4.7004609, 16.3528042, -5.6578388, 18.9135571, -23.6140175, 22.0106392
1: -7.8031425, 16.6316757, -9.2717447, 19.2887726, -27.0919151, 25.9034195
2: -6.3071837, 18.0582428, -7.5918212, 20.8814888, -27.1886730, 25.6500645
3: -6.9494028, 24.8218422, -8.2086658, 28.6347141, -35.5841064, 33.0305099
4: -5.5613313, 23.3349495, -6.6798763, 27.0115948, -32.5729256, 30.0148201

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B2_A2_B1

### Relational analysis result of IS_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7553153, upper bound: 27.7744902
time: 0.75 seconds

## Relational analysis of IS_A1_A1_B2_A2_B2

### Relational analysis result of IS_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7553153, upper bound: 27.7744902
time: 1.01 seconds

## BFS IS instance: IS_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -6.0793438, 20.2658539, -4.2426796, 15.0208941, -21.1002369, 24.5085335
1: -9.9371090, 20.6795712, -7.0561638, 15.2701111, -25.2072201, 27.7357349
2: -8.1595306, 22.3575153, -5.6916437, 16.6246243, -24.7841511, 28.0491600
3: -8.7953787, 30.6234856, -6.2818270, 22.8508835, -31.6462574, 36.9053078
4: -7.1967745, 28.9361477, -5.0319147, 21.4414806, -28.6382561, 33.9680595

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7748354, upper bound: 27.7767950
time: 0.91 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7748354, upper bound: 27.7845549
time: 0.85 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 5.33 seconds
IS_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.33
Output dim: 0, lower bound: -27.7553153, upper bound: 27.7748354
IS_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.33
Output dim: 0, lower bound: -27.7553153, upper bound: 27.7748354
IS_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.33
Output dim: 0, lower bound: -27.7553153, upper bound: 27.7744902
IS_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.33
Output dim: 0, lower bound: -27.7553153, upper bound: 27.7744902
IS_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.33
Output dim: 0, lower bound: -27.7748354, upper bound: 27.7767950
IS_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.33
Output dim: 0, lower bound: -27.7748354, upper bound: 27.7845549

## BFS IS instance: IS_A1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -4.4955974, 15.8341436, -5.5035720, 18.4324265, -22.9280243, 21.3377151
1: -7.4680882, 16.1045341, -9.0243711, 18.7931309, -26.2612190, 25.1289043
2: -6.0293436, 17.5095921, -7.3829780, 20.3584766, -26.3878193, 24.8925667
3: -6.6484203, 24.0976467, -7.9895844, 27.9114895, -34.5599098, 32.0872307
4: -5.3203859, 22.6204567, -6.4939799, 26.3111610, -31.6315460, 29.1144371

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7595153, upper bound: 27.7576097
time: 0.69 seconds

## Relational analysis of IS_A1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7532083, upper bound: 27.7500761
time: 1.15 seconds

## BFS IS instance: IS_A1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4.4955974, 15.8341436, -7.2312088, 22.6415482, -27.1371460, 23.0653534
1: -7.4680882, 16.1045341, -11.6248512, 23.1976414, -30.6657295, 27.7293835
2: -6.0293436, 17.5095921, -9.6295147, 24.9656944, -30.9950314, 27.1391068
3: -6.6484203, 24.0976467, -10.2161989, 34.2186089, -40.8670273, 34.3138466
4: -5.3203859, 22.6204567, -8.4509706, 32.4044380, -37.7248230, 31.0714264

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7595153, upper bound: 27.7576097
time: 0.66 seconds

## Relational analysis of IS_A1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.6989758, upper bound: 27.7500761
time: 0.70 seconds

## BFS IS instance: IS_A1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4.7004609, 16.3528042, -5.5035720, 18.4324265, -23.1328869, 21.8563766
1: -7.8031425, 16.6316757, -9.0243711, 18.7931309, -26.5962734, 25.6560440
2: -6.3071837, 18.0582428, -7.3829780, 20.3584766, -26.6656590, 25.4412193
3: -6.9494028, 24.8218422, -7.9895844, 27.9114895, -34.8608932, 32.8114281
4: -5.5613313, 23.3349495, -6.4939799, 26.3111610, -31.8724861, 29.8289299

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7468375, upper bound: 27.7509987
time: 0.71 seconds

## Relational analysis of IS_A1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7607951, upper bound: 27.7607880
time: 0.76 seconds

## BFS IS instance: IS_A1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4.7004609, 16.3528042, -7.2312088, 22.6415482, -27.3420086, 23.5840130
1: -7.8031425, 16.6316757, -11.6248512, 23.1976414, -31.0007839, 28.2565231
2: -6.3071837, 18.0582428, -9.6295147, 24.9656944, -31.2728786, 27.6877575
3: -6.9494028, 24.8218422, -10.2161989, 34.2186089, -41.1680031, 35.0380363
4: -5.5613313, 23.3349495, -8.4509706, 32.4044380, -37.9657707, 31.7859192

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_A2_B2_B1

### Relational analysis result of IS_A1_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7750984, upper bound: 27.7730287
time: 0.71 seconds

## Relational analysis of IS_A1_A1_B2_A2_B2_B2

### Relational analysis result of IS_A1_A1_B2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7607956, upper bound: 27.7607880
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6.0793438, 20.2658539, -4.4955974, 15.8341436, -21.9134865, 24.7614517
1: -9.9371090, 20.6795712, -7.4680882, 16.1045341, -26.0416412, 28.1476593
2: -8.1595306, 22.3575153, -6.0293436, 17.5095921, -25.6691189, 28.3868599
3: -8.7953787, 30.6234856, -6.6484203, 24.0976467, -32.8930244, 37.2719002
4: -7.1967745, 28.9361477, -5.3203859, 22.6204567, -29.8172302, 34.2565346

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7748354, upper bound: 27.7767950
time: 0.75 seconds

## Relational analysis of IS_A2_B1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_B1_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7576097, upper bound: 27.7595153
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B1_A2_B1_B2

### Relational analysis result of IS_A2_B1_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7500761, upper bound: 27.7532082
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6.0793438, 20.2658539, -4.7004609, 16.3528042, -22.4321480, 24.9663143
1: -9.9371090, 20.6795712, -7.8031425, 16.6316757, -26.5687828, 28.4827137
2: -8.1595306, 22.3575153, -6.3071837, 18.0582428, -26.2177715, 28.6646996
3: -8.7953787, 30.6234856, -6.9494028, 24.8218422, -33.6172218, 37.5728760
4: -7.1967745, 28.9361477, -5.5613313, 23.3349495, -30.5317154, 34.4974785

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7748354, upper bound: 27.7845549
time: 1.04 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7748354, upper bound: 27.7845549
time: 0.81 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 7.07 seconds
IS_A1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 6, time: 7.07
Output dim: 0, lower bound: -27.7595153, upper bound: 27.7576097
IS_A1_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 6, time: 7.07
Output dim: 0, lower bound: -27.7532083, upper bound: 27.7500761
IS_A1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 6, time: 7.07
Output dim: 0, lower bound: -27.7595153, upper bound: 27.7576097
IS_A1_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 6, time: 7.07
Output dim: 0, lower bound: -27.6989758, upper bound: 27.7500761
IS_A1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 7.07
Output dim: 0, lower bound: -27.7468375, upper bound: 27.7509987
IS_A1_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 6, time: 7.07
Output dim: 0, lower bound: -27.7607951, upper bound: 27.7607880
IS_A1_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -27.7750984, upper bound: 27.7730287
IS_A1_A1_B2_A2_B2_B2, status: Status.VERIFIED, split count: 6, time: 7.07
Output dim: 0, lower bound: -27.7607956, upper bound: 27.7607880
IS_A2_B1_B1_A2_B1_B1, status: Status.VERIFIED, split count: 6, time: 7.07
Output dim: 0, lower bound: -27.7576097, upper bound: 27.7595153
IS_A2_B1_B1_A2_B1_B2, status: Status.VERIFIED, split count: 6, time: 7.07
Output dim: 0, lower bound: -27.7500761, upper bound: 27.7532082
IS_A2_B1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -27.7748354, upper bound: 27.7845549
IS_A2_B1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -27.7748354, upper bound: 27.7845549

## BFS IS instance: IS_A1_A1_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -4.6756282, 16.2743206, -7.1337280, 22.3469582, -27.0225868, 23.4080448
1: -7.7626457, 16.5510101, -11.4693317, 22.8945293, -30.6571732, 28.0203400
2: -6.2732430, 17.9723988, -9.4977207, 24.6436653, -30.9169083, 27.4701195
3: -6.9134054, 24.7024574, -10.0791988, 33.7767944, -40.6902008, 34.7816544
4: -5.5317426, 23.2191429, -8.3362989, 31.9777317, -37.5094681, 31.5554428

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_A1_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_A1_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_A1_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_A1_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_A1_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_A1_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_A1_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_A2_B2_B1_A1

### Relational analysis result of IS_A1_A1_B2_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7468375, upper bound: 27.7509985
time: 1.32 seconds

## Relational analysis of IS_A1_A1_B2_A2_B2_B1_A2

### Relational analysis result of IS_A1_A1_B2_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7468375, upper bound: 27.7607880
time: 0.85 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -5.7951403, 19.3845100, -4.7004609, 16.3528042, -22.1479416, 24.0849705
1: -9.4860306, 19.7728939, -7.8031425, 16.6316757, -26.1177044, 27.5760365
2: -7.7768764, 21.4003086, -6.3071837, 18.0582428, -25.8351192, 27.7074909
3: -8.3971539, 29.3123341, -6.9494028, 24.8218422, -33.2189903, 36.2617378
4: -6.8590794, 27.6703186, -5.5613313, 23.3349495, -30.1940289, 33.2316475

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7707328, upper bound: 27.7710791
time: 1.01 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7704282, upper bound: 27.7703952
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -7.6980271, 24.1949196, -4.7004609, 16.3528042, -24.0508308, 28.8953800
1: -12.3796749, 24.7970486, -7.8031425, 16.6316757, -29.0113411, 32.6001892
2: -10.2556095, 26.6462612, -6.3071837, 18.0582428, -28.3138523, 32.9534454
3: -10.8816719, 36.5377197, -6.9494028, 24.8218422, -35.7035141, 43.4871216
4: -8.9968672, 34.6487350, -5.5613313, 23.3349495, -32.3318176, 40.2100677

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A1

### Relational analysis result of IS_A2_B1_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7755682, upper bound: 27.7777428
time: 0.91 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A2

### Relational analysis result of IS_A2_B1_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7704282, upper bound: 27.7703952
time: 0.83 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 10.28 seconds
IS_A1_A1_B2_A2_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 10.28
Output dim: 0, lower bound: -27.7468375, upper bound: 27.7509985
IS_A1_A1_B2_A2_B2_B1_A2, status: Status.VERIFIED, split count: 7, time: 10.28
Output dim: 0, lower bound: -27.7468375, upper bound: 27.7607880
IS_A2_B1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 10.28
Output dim: 0, lower bound: -27.7707328, upper bound: 27.7710791
IS_A2_B1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 10.28
Output dim: 0, lower bound: -27.7704282, upper bound: 27.7703952
IS_A2_B1_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 10.28
Output dim: 0, lower bound: -27.7755682, upper bound: 27.7777428
IS_A2_B1_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 10.28
Output dim: 0, lower bound: -27.7704282, upper bound: 27.7703952

## BFS IS instance: IS_A2_B1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -5.7641945, 19.2916203, -4.6283979, 16.1249676, -21.8891582, 23.9200134
1: -9.4368362, 19.6779194, -7.6856132, 16.3975601, -25.8343906, 27.3635330
2: -7.7358818, 21.2986374, -6.2086606, 17.8090305, -25.5449123, 27.5072975
3: -8.3541660, 29.1728878, -6.8450069, 24.4752293, -32.8293953, 36.0178909
4: -6.8234425, 27.5356026, -5.4754486, 22.9988766, -29.8223171, 33.0110474

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7856050, upper bound: 27.7837327
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7856050, upper bound: 27.7837327
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -5.6706810, 18.9784622, -4.5815530, 15.9177856, -21.5884647, 23.5600147
1: -9.2863588, 19.3620720, -7.6072607, 16.1925945, -25.4789543, 26.9693336
2: -7.6116576, 20.9604568, -6.1445537, 17.5895462, -25.2012024, 27.1050091
3: -8.2222252, 28.6992016, -6.7768393, 24.1542473, -32.3764572, 35.4760399
4: -6.7148666, 27.0846481, -5.4189653, 22.7074680, -29.4223347, 32.5036125

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7691742, upper bound: 27.7837327
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7691742, upper bound: 27.7837327
time: 1.10 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -7.5825920, 23.8449268, -4.6756282, 16.2743206, -23.8569107, 28.5205536
1: -12.1957550, 24.4371414, -7.7626457, 16.5510101, -28.7467651, 32.1997833
2: -10.1001081, 26.2645721, -6.2732430, 17.9723988, -28.0725060, 32.5378151
3: -10.7203646, 36.0134735, -6.9134054, 24.7024574, -35.4228210, 42.9268799
4: -8.8618498, 34.1424675, -5.5317426, 23.2191429, -32.0809898, 39.6742096

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A1_B1

### Relational analysis result of IS_A2_B1_B1_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7704282, upper bound: 27.7703952
time: 1.25 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A1_B2

### Relational analysis result of IS_A2_B1_B1_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7704282, upper bound: 27.7703952
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -7.4629340, 23.4205666, -4.6063285, 16.0388813, -23.5018158, 28.0268955
1: -12.0006990, 24.0046806, -7.6491113, 16.3112183, -28.3119125, 31.6537914
2: -9.9388428, 25.8092690, -6.1794634, 17.7171288, -27.6559715, 31.9887314
3: -10.5490026, 35.3722382, -6.8124738, 24.3457394, -34.8947411, 42.1847076
4: -8.7199173, 33.5306396, -5.4489064, 22.8786049, -31.5985222, 38.9795418

Time for backsubstitution: 2.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 34
type: A, layer: 3, pos: 34
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 27
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 27
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 20
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 46
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 1

Time for candidate selection: 9.67 seconds

### Candidate
type: A, layer: 3, pos: 39

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A2_A1

### Relational analysis result of IS_A2_B1_B1_A2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7704282, upper bound: 27.7703952
time: 0.82 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 34

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 34

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 12

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 31

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A2_A1

### Relational analysis result of IS_A2_B1_B1_A2_B2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7071053, upper bound: 27.7031565
time: 0.88 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 3

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 3

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 45

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 12

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 40

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 47

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 48

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 35

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 48

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A2_A1

### Relational analysis result of IS_A2_B1_B1_A2_B2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7465020, upper bound: 27.7490245
time: 1.52 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A2_A2

### Relational analysis result of IS_A2_B1_B1_A2_B2_A2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7179143, upper bound: 27.7137859
time: 0.88 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 27.46 seconds
IS_A2_B1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 27.46
Output dim: 0, lower bound: -27.7856050, upper bound: 27.7837327
IS_A2_B1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 27.46
Output dim: 0, lower bound: -27.7856050, upper bound: 27.7837327
IS_A2_B1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 27.46
Output dim: 0, lower bound: -27.7691742, upper bound: 27.7837327
IS_A2_B1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 27.46
Output dim: 0, lower bound: -27.7691742, upper bound: 27.7837327
IS_A2_B1_B1_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 27.46
Output dim: 0, lower bound: -27.7704282, upper bound: 27.7703952
IS_A2_B1_B1_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 27.46
Output dim: 0, lower bound: -27.7704282, upper bound: 27.7703952
IS_A2_B1_B1_A2_B2_A2_A2_A1, status: Status.VERIFIED, split count: 8, time: 27.46
Output dim: 0, lower bound: -27.7465020, upper bound: 27.7490245
IS_A2_B1_B1_A2_B2_A2_A2_A2, status: Status.VERIFIED, split count: 8, time: 27.46
Output dim: 0, lower bound: -27.7179143, upper bound: 27.7137859

## BFS IS instance: IS_A2_B1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -5.7050543, 19.1141205, -4.6283979, 16.1249676, -21.8300209, 23.7425137
1: -9.3431530, 19.4964027, -7.6856132, 16.3975601, -25.7407093, 27.1820164
2: -7.6575027, 21.1045666, -6.2086606, 17.8090305, -25.4665318, 27.3132267
3: -8.2721643, 28.9063854, -6.8450069, 24.4752293, -32.7473946, 35.7513924
4: -6.7553849, 27.2787571, -5.4754486, 22.9988766, -29.7542591, 32.7542000

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 34
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 27
type: A, layer: 3, pos: 27
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 13

Time for candidate selection: 8.73 seconds

### Candidate
type: B, layer: 3, pos: 12

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 12

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7811377, upper bound: 27.7816354
time: 0.74 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7806683, upper bound: 27.7807150
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -5.6406951, 18.8145924, -4.6283979, 16.1249676, -21.7656612, 23.4429874
1: -9.2340069, 19.2059059, -7.6856132, 16.3975601, -25.6315670, 26.8915195
2: -7.5728989, 20.7903633, -6.2086606, 17.8090305, -25.3819294, 26.9990234
3: -8.1777573, 28.4431610, -6.8450069, 24.4752293, -32.6529846, 35.2881699
4: -6.6805844, 26.8537216, -5.4754486, 22.9988766, -29.6794605, 32.3291626

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 34
type: B, layer: 3, pos: 34
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 27
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 27
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 13

Time for candidate selection: 8.73 seconds

### Candidate
type: B, layer: 3, pos: 12

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 12

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7697698, upper bound: 27.7816354
time: 0.69 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7806683, upper bound: 27.7807150
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -5.7050543, 19.1141205, -4.5815530, 15.9177856, -21.6228409, 23.6956730
1: -9.3431530, 19.4964027, -7.6072607, 16.1925945, -25.5357456, 27.1036625
2: -7.6575027, 21.1045666, -6.1445537, 17.5895462, -25.2470436, 27.2491207
3: -8.2721643, 28.9063854, -6.7768393, 24.1542473, -32.4264107, 35.6832237
4: -6.7553849, 27.2787571, -5.4189653, 22.7074680, -29.4628525, 32.6977234

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 34
type: B, layer: 3, pos: 34
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 27
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 27
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 13

Time for candidate selection: 8.97 seconds

### Candidate
type: A, layer: 3, pos: 12

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 12

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7805216, upper bound: 27.7806407
time: 0.78 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7805663, upper bound: 27.7806407
time: 0.85 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -5.6406951, 18.8145924, -4.5815530, 15.9177856, -21.5584812, 23.3961449
1: -9.2340069, 19.2059059, -7.6072607, 16.1925945, -25.4266014, 26.8131676
2: -7.5728989, 20.7903633, -6.1445537, 17.5895462, -25.1624432, 26.9349174
3: -8.1777573, 28.4431610, -6.7768393, 24.1542473, -32.3319931, 35.2200012
4: -6.6805844, 26.8537216, -5.4189653, 22.7074680, -29.3880520, 32.2726860

Time for backsubstitution: 2.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=30.45956039428711
rel_dist={0: [-27.852403738376353, 27.852403738376353]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 1109.64 seconds
