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
execution time: IAR + LP analysis = 2.64 + 1.87 = 4.51 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -27.8527630, upper bound: 27.8527630


# Binary Search by BASE starts (time budget: 1195.49 seconds, max iter: 100)

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
Binary search time: 85.92 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 1109.57 seconds

## Binary search (step 0) starts
Candidate diff: 0.2500000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8519590, upper bound: 27.8474240
time: 0.81 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8466200, upper bound: 27.8466200
time: 0.76 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.80 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.80
Output dim: 0, lower bound: -27.8519590, upper bound: 27.8474240
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.80
Output dim: 0, lower bound: -27.8466200, upper bound: 27.8466200

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -4.8150167, 17.2120781, -6.8545303, 23.1371231, -27.9521351, 24.0666046
1: -7.9723043, 17.4862099, -11.2150383, 23.6669369, -31.6392403, 28.7012482
2: -6.4574275, 18.9246101, -9.2105131, 25.4773407, -31.9347687, 28.1351242
3: -7.0645366, 26.1020927, -9.9516983, 34.9607430, -42.0252800, 36.0537910
4: -5.6939859, 24.4059410, -8.1457119, 33.1314163, -38.8254013, 32.5516510

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8466200, upper bound: 27.8466200
time: 0.80 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8466200, upper bound: 27.8466200
time: 0.79 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -6.2154808, 21.2638779, -6.9769545, 23.4826069, -29.6980877, 28.2408295
1: -10.2115059, 21.7044182, -11.4056101, 24.0275745, -34.2390823, 33.1100273
2: -8.3589554, 23.4411602, -9.3722210, 25.8551235, -34.2140770, 32.8133774
3: -9.0649576, 32.1635170, -10.1196079, 35.4739799, -44.5389290, 42.2831268
4: -7.4015422, 30.4624748, -8.2898979, 33.6252289, -41.0267715, 38.7523727

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8466200, upper bound: 27.8466200
time: 0.74 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8466200, upper bound: 27.8466200
time: 0.83 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.28 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.28
Output dim: 0, lower bound: -27.8466200, upper bound: 27.8466200
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.28
Output dim: 0, lower bound: -27.8466200, upper bound: 27.8466200
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.28
Output dim: 0, lower bound: -27.8466200, upper bound: 27.8466200
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.28
Output dim: 0, lower bound: -27.8466200, upper bound: 27.8466200

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -4.8150167, 17.2120781, -4.8150167, 17.2120781, -22.0270901, 22.0270901
1: -7.9723043, 17.4862099, -7.9723043, 17.4862099, -25.4585152, 25.4585152
2: -6.4574275, 18.9246101, -6.4574275, 18.9246101, -25.3820381, 25.3820381
3: -7.0645366, 26.1020927, -7.0645366, 26.1020927, -33.1666298, 33.1666298
4: -5.6939859, 24.4059410, -5.6939859, 24.4059410, -30.0999260, 30.0999260

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8022905, upper bound: 27.8075578
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
time: 0.74 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -4.8150167, 17.2120781, -6.2154808, 21.2638779, -26.0788898, 23.4275589
1: -7.9723043, 17.4862099, -10.2115059, 21.7044182, -29.6767235, 27.6977158
2: -6.4574275, 18.9246101, -8.3589554, 23.4411602, -29.8985863, 27.2835655
3: -7.0645366, 26.1020927, -9.0649576, 32.1635170, -39.2280540, 35.1670456
4: -5.6939859, 24.4059410, -7.4015422, 30.4624748, -36.1564598, 31.8074837

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8022905, upper bound: 27.8075578
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
time: 0.91 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -6.2154808, 21.2638779, -4.8150167, 17.2120781, -23.4275589, 26.0788918
1: -10.2115059, 21.7044182, -7.9723043, 17.4862099, -27.6977158, 29.6767235
2: -8.3589554, 23.4411602, -6.4574275, 18.9246101, -27.2835655, 29.8985825
3: -9.0649576, 32.1635170, -7.0645366, 26.1020927, -35.1670456, 39.2280540
4: -7.4015422, 30.4624748, -5.6939859, 24.4059410, -31.8074837, 36.1564598

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7969646, upper bound: 27.8067137
time: 1.08 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7601637, upper bound: 27.7601637
time: 0.68 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -6.2154808, 21.2638779, -6.2154808, 21.2638779, -27.4793587, 27.4793587
1: -10.2115059, 21.7044182, -10.2115059, 21.7044182, -31.9159241, 31.9159241
2: -8.3589554, 23.4411602, -8.3589554, 23.4411602, -31.8001156, 31.8001156
3: -9.0649576, 32.1635170, -9.0649576, 32.1635170, -41.2284698, 41.2284698
4: -7.4015422, 30.4624748, -7.4015422, 30.4624748, -37.8640175, 37.8640175

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7969646, upper bound: 27.8067137
time: 1.14 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7601637, upper bound: 27.7601637
time: 0.72 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.64 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.64
Output dim: 0, lower bound: -27.8022905, upper bound: 27.8075578
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.64
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.64
Output dim: 0, lower bound: -27.8022905, upper bound: 27.8075578
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.64
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.64
Output dim: 0, lower bound: -27.7969646, upper bound: 27.8067137
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 4.64
Output dim: 0, lower bound: -27.7601637, upper bound: 27.7601637
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.64
Output dim: 0, lower bound: -27.7969646, upper bound: 27.8067137
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 4.64
Output dim: 0, lower bound: -27.7601637, upper bound: 27.7601637

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4.1239586, 15.1485586, -4.8150167, 17.2120781, -21.3360367, 19.9635658
1: -6.8710823, 15.3317356, -7.9723043, 17.4862099, -24.3572922, 23.3040390
2: -5.5255399, 16.6497955, -6.4574275, 18.9246101, -24.4501495, 23.1072235
3: -6.0924091, 22.9905643, -7.0645366, 26.1020927, -32.1945038, 30.0550919
4: -4.8740869, 21.3502178, -5.6939859, 24.4059410, -29.2800274, 27.0442047

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8045752, upper bound: 27.8045752
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8045752, upper bound: 27.8045752
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -5.2290773, 18.9489193, -4.7824335, 17.1124096, -22.3414860, 23.7313519
1: -8.6279545, 19.1228676, -7.9202290, 17.3822308, -26.0101852, 27.0430965
2: -6.9899049, 20.7296162, -6.4132576, 18.8146172, -25.8045197, 27.1428719
3: -7.5971055, 28.6222916, -7.0183516, 25.9505577, -33.5476570, 35.6406403
4: -6.1555519, 26.5859127, -5.6552057, 24.2595177, -30.4150639, 32.2411194

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8045752, upper bound: 27.8045752
time: 1.26 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8045752, upper bound: 27.8045752
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4.1239586, 15.1485586, -6.2154808, 21.2638779, -25.3878365, 21.3640366
1: -6.8710823, 15.3317356, -10.2115059, 21.7044182, -28.5755005, 25.5432415
2: -5.5255399, 16.6497955, -8.3589554, 23.4411602, -28.9666996, 25.0087509
3: -6.0924091, 22.9905643, -9.0649576, 32.1635170, -38.2559280, 32.0555153
4: -4.8740869, 21.3502178, -7.4015422, 30.4624748, -35.3365555, 28.7517586

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -5.2290773, 18.9489193, -6.1691804, 21.1346111, -26.3636875, 25.1180992
1: -8.6279545, 19.1228676, -10.1390123, 21.5692558, -30.1972046, 29.2618790
2: -6.9899049, 20.7296162, -8.2967033, 23.2997894, -30.2896919, 29.0263195
3: -7.5971055, 28.6222916, -9.0015230, 31.9700451, -39.5671463, 37.6238136
4: -6.1555519, 26.5859127, -7.3470321, 30.2759399, -36.4314880, 33.9329453

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -5.4554358, 18.9849701, -4.8150167, 17.2120781, -22.6675148, 23.7999821
1: -9.0068264, 19.3360481, -7.9723043, 17.4862099, -26.4930363, 27.3083534
2: -7.3360786, 20.9468498, -6.4574275, 18.9246101, -26.2606888, 27.4042778
3: -8.0077810, 28.7492008, -7.0645366, 26.1020927, -34.1098709, 35.8137360
4: -6.5097780, 27.1807137, -5.6939859, 24.4059410, -30.9157124, 32.8746986

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7829412, upper bound: 27.7817977
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7829412, upper bound: 27.7817977
time: 1.08 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -5.4554358, 18.9849701, -6.2154808, 21.2638779, -26.7193127, 25.2004509
1: -9.0068264, 19.3360481, -10.2115059, 21.7044182, -30.7112446, 29.5475540
2: -7.3360786, 20.9468498, -8.3589554, 23.4411602, -30.7772388, 29.3058052
3: -8.0077810, 28.7492008, -9.0649576, 32.1635170, -40.1712990, 37.8141594
4: -6.5097780, 27.1807137, -7.4015422, 30.4624748, -36.9722519, 34.5822563

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7601637, upper bound: 27.7601637
time: 0.96 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7601637, upper bound: 27.7601637
time: 0.73 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.38 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.38
Output dim: 0, lower bound: -27.8045752, upper bound: 27.8045752
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.38
Output dim: 0, lower bound: -27.8045752, upper bound: 27.8045752
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.38
Output dim: 0, lower bound: -27.8045752, upper bound: 27.8045752
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.38
Output dim: 0, lower bound: -27.8045752, upper bound: 27.8045752
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.38
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.38
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.38
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.38
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.38
Output dim: 0, lower bound: -27.7829412, upper bound: 27.7817977
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.38
Output dim: 0, lower bound: -27.7829412, upper bound: 27.7817977
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 4.38
Output dim: 0, lower bound: -27.7601637, upper bound: 27.7601637
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 4.38
Output dim: 0, lower bound: -27.7601637, upper bound: 27.7601637

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -4.1239586, 15.1485586, -4.1239586, 15.1485586, -19.2725143, 19.2725124
1: -6.8710823, 15.3317356, -6.8710823, 15.3317356, -22.2028179, 22.2028179
2: -5.5255399, 16.6497955, -5.5255399, 16.6497955, -22.1753349, 22.1753349
3: -6.0924091, 22.9905643, -6.0924091, 22.9905643, -29.0829735, 29.0829735
4: -4.8740869, 21.3502178, -4.8740869, 21.3502178, -26.2243042, 26.2243042

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8253721, upper bound: 27.8295944
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8248058, upper bound: 27.8295960
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -4.1239586, 15.1485586, -5.2290773, 18.9489193, -23.0728779, 20.3776283
1: -6.8710823, 15.3317356, -8.6279545, 19.1228676, -25.9939499, 23.9596901
2: -5.5255399, 16.6497955, -6.9899049, 20.7296162, -26.2551556, 23.6396961
3: -6.0924091, 22.9905643, -7.5971055, 28.6222916, -34.7146988, 30.5876637
4: -4.8740869, 21.3502178, -6.1555519, 26.5859127, -31.4599972, 27.5057697

Time for backsubstitution: 2.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8253721, upper bound: 27.8295944
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8248058, upper bound: 27.8295960
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -5.2290773, 18.9489193, -4.1239586, 15.1485586, -20.3776302, 23.0728779
1: -8.6279545, 19.1228676, -6.8710823, 15.3317356, -23.9596901, 25.9939499
2: -6.9899049, 20.7296162, -5.5255399, 16.6497955, -23.6396961, 26.2551556
3: -7.5971055, 28.6222916, -6.0924091, 22.9905643, -30.5876656, 34.7146988
4: -6.1555519, 26.5859127, -4.8740869, 21.3502178, -27.5057697, 31.4599972

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7227716, upper bound: 27.7157986
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8045752, upper bound: 27.8045752
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -5.2290773, 18.9489193, -5.2290773, 18.9489193, -24.1779938, 24.1779938
1: -8.6279545, 19.1228676, -8.6279545, 19.1228676, -27.7508221, 27.7508221
2: -6.9899049, 20.7296162, -6.9899049, 20.7296162, -27.7195187, 27.7195187
3: -7.5971055, 28.6222916, -7.5971055, 28.6222916, -36.2193947, 36.2193947
4: -6.1555519, 26.5859127, -6.1555519, 26.5859127, -32.7414589, 32.7414589

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7227716, upper bound: 27.7157986
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8045752, upper bound: 27.8045752
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -4.1239586, 15.1485586, -5.4554358, 18.9849701, -23.1089249, 20.6039906
1: -6.8710823, 15.3317356, -9.0068264, 19.3360481, -26.2071304, 24.3385620
2: -5.5255399, 16.6497955, -7.3360786, 20.9468498, -26.4723892, 23.9858723
3: -6.0924091, 22.9905643, -8.0077810, 28.7492008, -34.8416100, 30.9983368
4: -4.8740869, 21.3502178, -6.5097780, 27.1807137, -32.0547943, 27.8599930

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.5979652, upper bound: 27.5601792
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8022905, upper bound: 27.8075578
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8017238, upper bound: 27.8075578
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4.1239586, 15.1485586, -6.3066802, 22.0238647, -26.1478233, 21.4552383
1: -6.8710823, 15.3317356, -10.3423834, 22.3369846, -29.2080669, 25.6741180
2: -5.5255399, 16.6497955, -8.4549847, 24.1596050, -29.6851425, 25.1047802
3: -6.0924091, 22.9905643, -9.1524858, 33.1810226, -39.2734299, 32.1430473
4: -4.8740869, 21.3502178, -7.4809628, 31.2509594, -36.1250381, 28.8311806

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.5979652, upper bound: 27.5601792
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8022905, upper bound: 27.8075578
time: 1.17 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8017238, upper bound: 27.8075578
time: 1.38 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -5.2290773, 18.9489193, -5.4554358, 18.9849701, -24.2140427, 24.4043541
1: -8.6279545, 19.1228676, -9.0068264, 19.3360481, -27.9640026, 28.1296940
2: -6.9899049, 20.7296162, -7.3360786, 20.9468498, -27.9367523, 28.0656948
3: -7.5971055, 28.6222916, -8.0077810, 28.7492008, -36.3463058, 36.6300621
4: -6.1555519, 26.5859127, -6.5097780, 27.1807137, -33.3362617, 33.0956841

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.4554471, upper bound: 27.4108028
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.6999942, upper bound: 27.6941646
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -5.2290773, 18.9489193, -6.3066802, 22.0238647, -27.2529373, 25.2556000
1: -8.6279545, 19.1228676, -10.3423834, 22.3369846, -30.9649391, 29.4652519
2: -6.9899049, 20.7296162, -8.4549847, 24.1596050, -31.1495094, 29.1846008
3: -7.5971055, 28.6222916, -9.1524858, 33.1810226, -40.7781219, 37.7747765
4: -6.1555519, 26.5859127, -7.4809628, 31.2509594, -37.4065094, 34.0668755

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.4554471, upper bound: 27.4108028
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.6999942, upper bound: 27.6941646
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -5.4554358, 18.9849701, -4.1239586, 15.1485586, -20.6039886, 23.1089268
1: -9.0068264, 19.3360481, -6.8710823, 15.3317356, -24.3385620, 26.2071304
2: -7.3360786, 20.9468498, -5.5255399, 16.6497955, -23.9858723, 26.4723892
3: -8.0077810, 28.7492008, -6.0924091, 22.9905643, -30.9983368, 34.8416100
4: -6.5097780, 27.1807137, -4.8740869, 21.3502178, -27.8599949, 32.0547943

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7131793, upper bound: 27.7411855
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8200546, upper bound: 27.8287580
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -5.4554358, 18.9849701, -5.2290773, 18.9489193, -24.4043541, 24.2140408
1: -9.0068264, 19.3360481, -8.6279545, 19.1228676, -28.1296940, 27.9640026
2: -7.3360786, 20.9468498, -6.9899049, 20.7296162, -28.0656948, 27.9367523
3: -8.0077810, 28.7492008, -7.5971055, 28.6222916, -36.6300621, 36.3463058
4: -6.5097780, 27.1807137, -6.1555519, 26.5859127, -33.0956841, 33.3362617

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7131793, upper bound: 27.7411855
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8200546, upper bound: 27.8287580
time: 0.68 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 5.53 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.53
Output dim: 0, lower bound: -27.8253721, upper bound: 27.8295944
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.53
Output dim: 0, lower bound: -27.8248058, upper bound: 27.8295960
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.53
Output dim: 0, lower bound: -27.8253721, upper bound: 27.8295944
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.53
Output dim: 0, lower bound: -27.8248058, upper bound: 27.8295960
IS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.53
Output dim: 0, lower bound: -27.7227716, upper bound: 27.7157986
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.53
Output dim: 0, lower bound: -27.8045752, upper bound: 27.8045752
IS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 5.53
Output dim: 0, lower bound: -27.7227716, upper bound: 27.7157986
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.53
Output dim: 0, lower bound: -27.8045752, upper bound: 27.8045752
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.53
Output dim: 0, lower bound: -27.8022905, upper bound: 27.8075578
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.53
Output dim: 0, lower bound: -27.8017238, upper bound: 27.8075578
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.53
Output dim: 0, lower bound: -27.8022905, upper bound: 27.8075578
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.53
Output dim: 0, lower bound: -27.8017238, upper bound: 27.8075578
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.53
Output dim: 0, lower bound: -27.6999942, upper bound: 27.6941646
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.53
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 5.53
Output dim: 0, lower bound: -27.6999942, upper bound: 27.6941646
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.53
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
IS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.53
Output dim: 0, lower bound: -27.7131793, upper bound: 27.7411855
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.53
Output dim: 0, lower bound: -27.8200546, upper bound: 27.8287580
IS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 5.53
Output dim: 0, lower bound: -27.7131793, upper bound: 27.7411855
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.53
Output dim: 0, lower bound: -27.8200546, upper bound: 27.8287580

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -3.9672580, 14.7736979, -4.0972376, 15.0651321, -19.0323906, 18.8709316
1: -6.6120090, 14.9280148, -6.8283501, 15.2466536, -21.8586617, 21.7563648
2: -5.3131099, 16.2373619, -5.4895091, 16.5589733, -21.8720837, 21.7268715
3: -5.8528652, 22.4279957, -6.0548358, 22.8659801, -28.7188454, 28.4828320
4: -4.6880665, 20.7485371, -4.8424792, 21.2284870, -25.9165535, 25.5910168

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8521223, upper bound: 27.8521223
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8521223, upper bound: 27.8521223
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -4.0236449, 14.8443575, -4.1239586, 15.1485586, -19.1721935, 18.9683132
1: -6.7107162, 15.0167494, -6.8710823, 15.3317356, -22.0424519, 21.8878307
2: -5.3908873, 16.3171444, -5.5255399, 16.6497955, -22.0406837, 21.8426838
3: -5.9503188, 22.5346546, -6.0924091, 22.9905643, -28.9408836, 28.6270638
4: -4.7566547, 20.9033737, -4.8740869, 21.3502178, -26.1068726, 25.7774601

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8521223, upper bound: 27.8521345
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8521223, upper bound: 27.8521345
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -3.9672580, 14.7736979, -5.1997471, 18.8553143, -22.8225727, 19.9734459
1: -6.6120090, 14.9280148, -8.5812101, 19.0277405, -25.6397495, 23.5092239
2: -5.3131099, 16.2373619, -6.9503589, 20.6281948, -25.9413052, 23.1877213
3: -5.8528652, 22.4279957, -7.5556221, 28.4825630, -34.3354263, 29.9836121
4: -4.6880665, 20.7485371, -6.1209035, 26.4516563, -31.1397228, 26.8694401

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7369988, upper bound: 27.7491186
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7369988, upper bound: 27.8295944
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -4.0236449, 14.8443575, -5.2290773, 18.9489193, -22.9725590, 20.0734291
1: -6.7107162, 15.0167494, -8.6279545, 19.1228676, -25.8335838, 23.6447029
2: -5.3908873, 16.3171444, -6.9899049, 20.7296162, -26.1205025, 23.3070469
3: -5.9503188, 22.5346546, -7.5971055, 28.6222916, -34.5726089, 30.1317596
4: -4.7566547, 20.9033737, -6.1555519, 26.5859127, -31.3425655, 27.0589256

Time for backsubstitution: 2.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7369988, upper bound: 27.7491308
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7369988, upper bound: 27.8295960
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -5.1221609, 18.6184425, -4.1239586, 15.1485586, -20.2707195, 22.7424011
1: -8.4564943, 18.7807064, -6.8710823, 15.3317356, -23.7882309, 25.6517887
2: -6.8467531, 20.3681755, -5.5255399, 16.6497955, -23.4965458, 25.8937149
3: -7.4440994, 28.1251602, -6.0924091, 22.9905643, -30.4346581, 34.2175674
4: -6.0290127, 26.1052856, -4.8740869, 21.3502178, -27.3792305, 30.9793720

Time for backsubstitution: 2.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8295944, upper bound: 27.8248058
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8295944, upper bound: 27.8248058
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -5.1221609, 18.6184425, -5.2290773, 18.9489193, -24.0710793, 23.8475170
1: -8.4564943, 18.7807064, -8.6279545, 19.1228676, -27.5793610, 27.4086609
2: -6.8467531, 20.3681755, -6.9899049, 20.7296162, -27.5763683, 27.3580780
3: -7.4440994, 28.1251602, -7.5971055, 28.6222916, -36.0663834, 35.7222633
4: -6.0290127, 26.1052856, -6.1555519, 26.5859127, -32.6149254, 32.2608376

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7157986, upper bound: 27.7227716
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7157986, upper bound: 27.8045752
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -3.9672580, 14.7736979, -5.4248290, 18.8884392, -22.8556976, 20.1985245
1: -6.6120090, 14.9280148, -8.9576893, 19.2364769, -25.8484859, 23.8857040
2: -5.3131099, 16.2373619, -7.2945261, 20.8414860, -26.1545963, 23.5318871
3: -5.8528652, 22.4279957, -7.9642119, 28.6051235, -34.4579887, 30.3922005
4: -4.6880665, 20.7485371, -6.4730754, 27.0406895, -31.7287502, 27.2216129

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8068519, upper bound: 27.7901619
time: 1.28 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8068519, upper bound: 27.8474118
time: 1.09 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -4.0236449, 14.8443575, -5.4554358, 18.9849701, -23.0086060, 20.2997913
1: -6.7107162, 15.0167494, -9.0068264, 19.3360481, -26.0467625, 24.0235748
2: -5.3908873, 16.3171444, -7.3360786, 20.9468498, -26.3377380, 23.6532230
3: -5.9503188, 22.5346546, -8.0077810, 28.7492008, -34.6995201, 30.5424309
4: -4.7566547, 20.9033737, -6.5097780, 27.1807137, -31.9373665, 27.4131508

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8068519, upper bound: 27.7901787
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8068519, upper bound: 27.8474240
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -3.9672580, 14.7736979, -6.2734494, 21.9204998, -25.8877583, 21.0471420
1: -6.6120090, 14.9280148, -10.2893724, 22.2297630, -28.8417721, 25.2173882
2: -5.3131099, 16.2373619, -8.4102154, 24.0473728, -29.3604832, 24.6475754
3: -5.8528652, 22.4279957, -9.1053514, 33.0274124, -38.8802795, 31.5333347
4: -4.6880665, 20.7485371, -7.4416513, 31.1024818, -35.7905502, 28.1901875

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.5423100, upper bound: 27.5845282
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.2412960, upper bound: 27.1541456
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -4.0236449, 14.8443575, -6.3066802, 22.0238647, -26.0475044, 21.1510372
1: -6.7107162, 15.0167494, -10.3423834, 22.3369846, -29.0476990, 25.3591309
2: -5.3908873, 16.3171444, -8.4549847, 24.1596050, -29.5504913, 24.7721291
3: -5.9503188, 22.5346546, -9.1524858, 33.1810226, -39.1313362, 31.6871414
4: -4.7566547, 20.9033737, -7.4809628, 31.2509594, -36.0076103, 28.3843365

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.5317944, upper bound: 27.5804342
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.2304645, upper bound: 27.1473526
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -5.1221609, 18.6184425, -5.4554358, 18.9849701, -24.1071320, 24.0738773
1: -8.4564943, 18.7807064, -9.0068264, 19.3360481, -27.7925415, 27.7875328
2: -6.8467531, 20.3681755, -7.3360786, 20.9468498, -27.7936020, 27.7042542
3: -7.4440994, 28.1251602, -8.0077810, 28.7492008, -36.1932983, 36.1329384
4: -6.0290127, 26.1052856, -6.5097780, 27.1807137, -33.2097244, 32.6150627

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7411855, upper bound: 27.7131793
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7411855, upper bound: 27.8200546
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -5.1221609, 18.6184425, -6.3066802, 22.0238647, -27.1460266, 24.9251232
1: -8.4564943, 18.7807064, -10.3423834, 22.3369846, -30.7934799, 29.1230888
2: -6.8467531, 20.3681755, -8.4549847, 24.1596050, -31.0063591, 28.8231602
3: -7.4440994, 28.1251602, -9.1524858, 33.1810226, -40.6251144, 37.2776451
4: -6.0290127, 26.1052856, -7.4809628, 31.2509594, -37.2799721, 33.5862503

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.5442157, upper bound: 27.5866045
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.2410869, upper bound: 27.1536559
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -5.3233452, 18.5752125, -4.1239586, 15.1485586, -20.4719009, 22.6991711
1: -8.7951794, 18.9134140, -6.8710823, 15.3317356, -24.1269150, 25.7844963
2: -7.1597281, 20.4994736, -5.5255399, 16.6497955, -23.8095188, 26.0250130
3: -7.8201966, 28.1347542, -6.0924091, 22.9905643, -30.8107529, 34.2271652
4: -6.3551607, 26.5873337, -4.8740869, 21.3502178, -27.7053795, 31.4614201

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8322060, upper bound: 27.8351593
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8474118, upper bound: 27.8513305
time: 0.91 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8474118, upper bound: 27.8513305
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -5.3233452, 18.5752125, -5.2290773, 18.9489193, -24.2722645, 23.8042870
1: -8.7951794, 18.9134140, -8.6279545, 19.1228676, -27.9180470, 27.5413685
2: -7.1597281, 20.4994736, -6.9899049, 20.7296162, -27.8893433, 27.4893761
3: -7.8201966, 28.1347542, -7.5971055, 28.6222916, -36.4424820, 35.7318611
4: -6.3551607, 26.5873337, -6.1555519, 26.5859127, -32.9410744, 32.7428856

Time for backsubstitution: 2.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8019506, upper bound: 27.8089876
time: 1.02 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7519013, upper bound: 27.7754321
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7322883, upper bound: 27.7483268
time: 1.08 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7322883, upper bound: 27.8287580
time: 0.59 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 7.24 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.24
Output dim: 0, lower bound: -27.8521223, upper bound: 27.8521223
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.24
Output dim: 0, lower bound: -27.8521223, upper bound: 27.8521223
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.24
Output dim: 0, lower bound: -27.8521223, upper bound: 27.8521345
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.24
Output dim: 0, lower bound: -27.8521223, upper bound: 27.8521345
IS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 7.24
Output dim: 0, lower bound: -27.7369988, upper bound: 27.7491186
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.24
Output dim: 0, lower bound: -27.7369988, upper bound: 27.8295944
IS_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 7.24
Output dim: 0, lower bound: -27.7369988, upper bound: 27.7491308
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.24
Output dim: 0, lower bound: -27.7369988, upper bound: 27.8295960
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.24
Output dim: 0, lower bound: -27.8295944, upper bound: 27.8248058
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.24
Output dim: 0, lower bound: -27.8295944, upper bound: 27.8248058
IS_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 7.24
Output dim: 0, lower bound: -27.7157986, upper bound: 27.7227716
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.24
Output dim: 0, lower bound: -27.7157986, upper bound: 27.8045752
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.24
Output dim: 0, lower bound: -27.8068519, upper bound: 27.7901619
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.24
Output dim: 0, lower bound: -27.8068519, upper bound: 27.8474118
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.24
Output dim: 0, lower bound: -27.8068519, upper bound: 27.7901787
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.24
Output dim: 0, lower bound: -27.8068519, upper bound: 27.8474240
IS_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 7.24
Output dim: 0, lower bound: -27.5423100, upper bound: 27.5845282
IS_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 7.24
Output dim: 0, lower bound: -27.2412960, upper bound: 27.1541456
IS_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 7.24
Output dim: 0, lower bound: -27.5317944, upper bound: 27.5804342
IS_A1_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 7.24
Output dim: 0, lower bound: -27.2304645, upper bound: 27.1473526
IS_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 7.24
Output dim: 0, lower bound: -27.7411855, upper bound: 27.7131793
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.24
Output dim: 0, lower bound: -27.7411855, upper bound: 27.8200546
IS_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 7.24
Output dim: 0, lower bound: -27.5442157, upper bound: 27.5866045
IS_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 7.24
Output dim: 0, lower bound: -27.2410869, upper bound: 27.1536559
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.24
Output dim: 0, lower bound: -27.8474118, upper bound: 27.8513305
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.24
Output dim: 0, lower bound: -27.8474118, upper bound: 27.8513305
IS_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 7.24
Output dim: 0, lower bound: -27.7322883, upper bound: 27.7483268
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.24
Output dim: 0, lower bound: -27.7322883, upper bound: 27.8287580

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -3.9672580, 14.7736979, -3.9672580, 14.7736979, -18.7409554, 18.7409554
1: -6.6120090, 14.9280148, -6.6120090, 14.9280148, -21.5400238, 21.5400238
2: -5.3131099, 16.2373619, -5.3131099, 16.2373619, -21.5504723, 21.5504723
3: -5.8528652, 22.4279957, -5.8528652, 22.4279957, -28.2808590, 28.2808590
4: -4.6880665, 20.7485371, -4.6880665, 20.7485371, -25.4366035, 25.4366035

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8512453, upper bound: 27.8510233
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8495177, upper bound: 27.8495177
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -3.9672580, 14.7736979, -4.0236449, 14.8443575, -18.8116131, 18.7973366
1: -6.6120090, 14.9280148, -6.7107162, 15.0167494, -21.6287575, 21.6387310
2: -5.3131099, 16.2373619, -5.3908873, 16.3171444, -21.6302547, 21.6282501
3: -5.8528652, 22.4279957, -5.9503188, 22.5346546, -28.3875179, 28.3783150
4: -4.6880665, 20.7485371, -4.7566547, 20.9033737, -25.5914402, 25.5051918

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8512453, upper bound: 27.8518782
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8495177, upper bound: 27.8503726
time: 1.27 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -4.0236449, 14.8443575, -3.9672580, 14.7736979, -18.7973366, 18.8116150
1: -6.7107162, 15.0167494, -6.6120090, 14.9280148, -21.6387310, 21.6287575
2: -5.3908873, 16.3171444, -5.3131099, 16.2373619, -21.6282501, 21.6302547
3: -5.9503188, 22.5346546, -5.8528652, 22.4279957, -28.3783150, 28.3875179
4: -4.7566547, 20.9033737, -4.6880665, 20.7485371, -25.5051918, 25.5914402

Time for backsubstitution: 2.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8506168, upper bound: 27.8510354
time: 1.30 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8503726, upper bound: 27.8510354
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -4.0236449, 14.8443575, -4.0236449, 14.8443575, -18.8679943, 18.8679943
1: -6.7107162, 15.0167494, -6.7107162, 15.0167494, -21.7274647, 21.7274628
2: -5.3908873, 16.3171444, -5.3908873, 16.3171444, -21.7080307, 21.7080307
3: -5.9503188, 22.5346546, -5.9503188, 22.5346546, -28.4849739, 28.4849739
4: -4.7566547, 20.9033737, -4.7566547, 20.9033737, -25.6600285, 25.6600285

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8506168, upper bound: 27.8518903
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8503726, upper bound: 27.8518903
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -3.9672580, 14.7736979, -5.1221609, 18.6184425, -22.5857010, 19.8958588
1: -6.6120090, 14.9280148, -8.4564943, 18.7807064, -25.3927155, 23.3845100
2: -5.3131099, 16.2373619, -6.8467531, 20.3681755, -25.6812859, 23.0841103
3: -5.8528652, 22.4279957, -7.4440994, 28.1251602, -33.9780273, 29.8720894
4: -4.6880665, 20.7485371, -6.0290127, 26.1052856, -30.7933521, 26.7775497

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7187981, upper bound: 27.8096160
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7376273, upper bound: 27.8295944
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4.0236449, 14.8443575, -5.1221609, 18.6184425, -22.6420841, 19.9665184
1: -6.7107162, 15.0167494, -8.4564943, 18.7807064, -25.4914207, 23.4732437
2: -5.3908873, 16.3171444, -6.8467531, 20.3681755, -25.7590637, 23.1638947
3: -5.9503188, 22.5346546, -7.4440994, 28.1251602, -34.0754776, 29.9787483
4: -4.7566547, 20.9033737, -6.0290127, 26.1052856, -30.8619404, 26.9323864

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.4972300, upper bound: 27.5499982
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7369988, upper bound: 27.8291396
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -5.1221609, 18.6184425, -3.9672580, 14.7736979, -19.8958588, 22.5857010
1: -8.4564943, 18.7807064, -6.6120090, 14.9280148, -23.3845100, 25.3927155
2: -6.8467531, 20.3681755, -5.3131099, 16.2373619, -23.0841103, 25.6812859
3: -7.4440994, 28.1251602, -5.8528652, 22.4279957, -29.8720913, 33.9780273
4: -6.0290127, 26.1052856, -4.6880665, 20.7485371, -26.7775497, 30.7933521

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7549919, upper bound: 27.7419981
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8294166, upper bound: 27.8246763
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -5.1221609, 18.6184425, -4.0236449, 14.8443575, -19.9665184, 22.6420822
1: -8.4564943, 18.7807064, -6.7107162, 15.0167494, -23.4732437, 25.4914207
2: -6.8467531, 20.3681755, -5.3908873, 16.3171444, -23.1638947, 25.7590637
3: -7.4440994, 28.1251602, -5.9503188, 22.5346546, -29.9787521, 34.0754776
4: -6.0290127, 26.1052856, -4.7566547, 20.9033737, -26.9323864, 30.8619404

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7549919, upper bound: 27.7423736
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8294166, upper bound: 27.8247772
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -5.1221609, 18.6184425, -5.1221609, 18.6184425, -23.7406044, 23.7406044
1: -8.4564943, 18.7807064, -8.4564943, 18.7807064, -27.2372017, 27.2371998
2: -6.8467531, 20.3681755, -6.8467531, 20.3681755, -27.2149277, 27.2149277
3: -7.4440994, 28.1251602, -7.4440994, 28.1251602, -35.5692558, 35.5692558
4: -6.0290127, 26.1052856, -6.0290127, 26.1052856, -32.1343002, 32.1343002

Time for backsubstitution: 2.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.4128059, upper bound: 27.4901681
time: 0.99 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7157986, upper bound: 27.8045752
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -3.9672580, 14.7736979, -5.3472810, 18.5768604, -22.5441189, 20.1209793
1: -6.6120090, 14.9280148, -8.8134174, 18.9273415, -25.5393505, 23.7414322
2: -5.3131099, 16.2373619, -7.1847754, 20.4710217, -25.7841320, 23.4221306
3: -5.8528652, 22.4279957, -7.8320279, 28.1110363, -33.9639015, 30.2600231
4: -4.6880665, 20.7485371, -6.3711629, 26.5981026, -31.2861691, 27.1196995

Time for backsubstitution: 2.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8075718, upper bound: 27.7901619
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8063275, upper bound: 27.7894296
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -3.9672580, 14.7736979, -5.3233452, 18.5752125, -22.5424709, 20.0970421
1: -6.6120090, 14.9280148, -8.7951794, 18.9134140, -25.5254230, 23.7231941
2: -5.3131099, 16.2373619, -7.1597281, 20.4994736, -25.8125839, 23.3970871
3: -5.8528652, 22.4279957, -7.8201966, 28.1347542, -33.9876175, 30.2481804
4: -4.6880665, 20.7485371, -6.3551607, 26.5873337, -31.2753983, 27.1036987

Time for backsubstitution: 2.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8075718, upper bound: 27.8474118
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8063275, upper bound: 27.8459063
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -4.0236449, 14.8443575, -5.3472810, 18.5768604, -22.6005020, 20.1916389
1: -6.7107162, 15.0167494, -8.8134174, 18.9273415, -25.6380577, 23.8301659
2: -5.3908873, 16.3171444, -7.1847754, 20.4710217, -25.8619080, 23.5019150
3: -5.9503188, 22.5346546, -7.8320279, 28.1110363, -34.0613556, 30.3666821
4: -4.7566547, 20.9033737, -6.3711629, 26.5981026, -31.3547573, 27.2745361

Time for backsubstitution: 2.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8068519, upper bound: 27.7901787
time: 1.08 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8067355, upper bound: 27.7901787
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -4.0236449, 14.8443575, -5.3233452, 18.5752125, -22.5988541, 20.1676998
1: -6.7107162, 15.0167494, -8.7951794, 18.9134140, -25.6241302, 23.8119278
2: -5.3908873, 16.3171444, -7.1597281, 20.4994736, -25.8903618, 23.4768715
3: -5.9503188, 22.5346546, -7.8201966, 28.1347542, -34.0850716, 30.3548431
4: -4.7566547, 20.9033737, -6.3551607, 26.5873337, -31.3439865, 27.2585335

Time for backsubstitution: 2.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8068519, upper bound: 27.7962482
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8063275, upper bound: 27.8466200
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -5.1221609, 18.6184425, -5.3233452, 18.5752125, -23.6973724, 23.9417877
1: -8.4564943, 18.7807064, -8.7951794, 18.9134140, -27.3699074, 27.5758839
2: -6.8467531, 20.3681755, -7.1597281, 20.4994736, -27.3462257, 27.5279007
3: -7.4440994, 28.1251602, -7.8201966, 28.1347542, -35.5788498, 35.9453583
4: -6.0290127, 26.1052856, -6.3551607, 26.5873337, -32.6163483, 32.4604454

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.6242533, upper bound: 27.6787513
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7411855, upper bound: 27.8039116
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -5.3233452, 18.5752125, -3.9672580, 14.7736979, -20.0970421, 22.5424709
1: -8.7951794, 18.9134140, -6.6120090, 14.9280148, -23.7231941, 25.5254230
2: -7.1597281, 20.4994736, -5.3131099, 16.2373619, -23.3970852, 25.8125839
3: -7.8201966, 28.1347542, -5.8528652, 22.4279957, -30.2481861, 33.9876175
4: -6.3551607, 26.5873337, -4.6880665, 20.7485371, -27.1036987, 31.2754002

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8392921, upper bound: 27.8452579
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8249007, upper bound: 27.8252235
time: 1.00 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -5.3233452, 18.5752125, -4.0236449, 14.8443575, -20.1676998, 22.5988541
1: -8.7951794, 18.9134140, -6.7107162, 15.0167494, -23.8119278, 25.6241302
2: -7.1597281, 20.4994736, -5.3908873, 16.3171444, -23.4768715, 25.8903618
3: -7.8201966, 28.1347542, -5.9503188, 22.5346546, -30.3548470, 34.0850716
4: -6.3551607, 26.5873337, -4.7566547, 20.9033737, -27.2585335, 31.3439865

Time for backsubstitution: 2.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8392921, upper bound: 27.8452579
time: 1.03 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8249007, upper bound: 27.8252235
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -5.3233452, 18.5752125, -5.1221609, 18.6184425, -23.9417877, 23.6973724
1: -8.7951794, 18.9134140, -8.4564943, 18.7807064, -27.5758839, 27.3699074
2: -7.1597281, 20.4994736, -6.8467531, 20.3681755, -27.5279007, 27.3462257
3: -7.8201966, 28.1347542, -7.4440994, 28.1251602, -35.9453583, 35.5788498
4: -6.3551607, 26.5873337, -6.0290127, 26.1052856, -32.4604454, 32.6163483

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.6370678, upper bound: 27.6072014
time: 1.09 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.1588498, upper bound: 27.2544260
time: 0.92 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 7.89 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 7.89
Output dim: 0, lower bound: -27.8512453, upper bound: 27.8510233
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 7.89
Output dim: 0, lower bound: -27.8495177, upper bound: 27.8495177
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 7.89
Output dim: 0, lower bound: -27.8512453, upper bound: 27.8518782
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 7.89
Output dim: 0, lower bound: -27.8495177, upper bound: 27.8503726
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 7.89
Output dim: 0, lower bound: -27.8506168, upper bound: 27.8510354
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 7.89
Output dim: 0, lower bound: -27.8503726, upper bound: 27.8510354
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 7.89
Output dim: 0, lower bound: -27.8506168, upper bound: 27.8518903
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 7.89
Output dim: 0, lower bound: -27.8503726, upper bound: 27.8518903
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 7.89
Output dim: 0, lower bound: -27.7187981, upper bound: 27.8096160
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 7.89
Output dim: 0, lower bound: -27.7376273, upper bound: 27.8295944
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 7.89
Output dim: 0, lower bound: -27.4972300, upper bound: 27.5499982
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 7.89
Output dim: 0, lower bound: -27.7369988, upper bound: 27.8291396
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 7.89
Output dim: 0, lower bound: -27.7549919, upper bound: 27.7419981
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 7.89
Output dim: 0, lower bound: -27.8294166, upper bound: 27.8246763
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 7.89
Output dim: 0, lower bound: -27.7549919, upper bound: 27.7423736
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 7.89
Output dim: 0, lower bound: -27.8294166, upper bound: 27.8247772
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 7.89
Output dim: 0, lower bound: -27.4128059, upper bound: 27.4901681
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 7.89
Output dim: 0, lower bound: -27.7157986, upper bound: 27.8045752
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 7.89
Output dim: 0, lower bound: -27.8075718, upper bound: 27.7901619
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 7.89
Output dim: 0, lower bound: -27.8063275, upper bound: 27.7894296
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 7.89
Output dim: 0, lower bound: -27.8075718, upper bound: 27.8474118
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 7.89
Output dim: 0, lower bound: -27.8063275, upper bound: 27.8459063
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 7.89
Output dim: 0, lower bound: -27.8068519, upper bound: 27.7901787
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 7.89
Output dim: 0, lower bound: -27.8067355, upper bound: 27.7901787
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 7.89
Output dim: 0, lower bound: -27.8068519, upper bound: 27.7962482
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 7.89
Output dim: 0, lower bound: -27.8063275, upper bound: 27.8466200
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 7.89
Output dim: 0, lower bound: -27.6242533, upper bound: 27.6787513
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 7.89
Output dim: 0, lower bound: -27.7411855, upper bound: 27.8039116
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 7.89
Output dim: 0, lower bound: -27.8392921, upper bound: 27.8452579
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 7.89
Output dim: 0, lower bound: -27.8249007, upper bound: 27.8252235
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 7.89
Output dim: 0, lower bound: -27.8392921, upper bound: 27.8452579
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 7.89
Output dim: 0, lower bound: -27.8249007, upper bound: 27.8252235
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 7.89
Output dim: 0, lower bound: -27.6370678, upper bound: 27.6072014
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 7.89
Output dim: 0, lower bound: -27.1588498, upper bound: 27.2544260

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -3.3273158, 12.8372402, -3.9184031, 14.6225538, -17.9498692, 16.7556438
1: -5.6021156, 12.9728231, -6.5344067, 14.7736359, -20.3757515, 19.5072250
2: -4.4462528, 14.1458454, -5.2468700, 16.0727081, -20.5189610, 19.3927155
3: -4.9587526, 19.5276680, -5.7845664, 22.2014503, -27.1602020, 25.3122349
4: -3.9472556, 18.0581570, -4.6306186, 20.5310402, -24.4782963, 22.6887760

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8495177, upper bound: 27.8495177
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8495177, upper bound: 27.8495177
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -3.8642793, 14.4800587, -3.9672580, 14.7736979, -18.6379757, 18.4473171
1: -6.4491496, 14.6237555, -6.6120090, 14.9280148, -21.3771648, 21.2357635
2: -5.1740036, 15.9146786, -5.3131099, 16.2373619, -21.4113617, 21.2277889
3: -5.7092237, 21.9910412, -5.8528652, 22.4279957, -28.1372185, 27.8439026
4: -4.5661373, 20.3150978, -4.6880665, 20.7485371, -25.3146744, 25.0031643

Time for backsubstitution: 2.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8495177, upper bound: 27.8495177
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8495177, upper bound: 27.8495177
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -3.3273158, 12.8372402, -3.9740572, 14.6925678, -18.0198841, 16.8112965
1: -5.6021156, 12.9728231, -6.6321397, 14.8614407, -20.4635563, 19.6049633
2: -4.4462528, 14.1458454, -5.3237147, 16.1516628, -20.5979156, 19.4695606
3: -4.9587526, 19.5276680, -5.8811302, 22.3070793, -27.2658291, 25.4087982
4: -3.9472556, 18.0581570, -4.6983848, 20.6851788, -24.6324348, 22.7565403

Time for backsubstitution: 2.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8510354, upper bound: 27.8503726
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8510354, upper bound: 27.8503726
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -3.8642793, 14.4800587, -4.0236449, 14.8443575, -18.7086334, 18.5037003
1: -6.4491496, 14.6237555, -6.7107162, 15.0167494, -21.4658985, 21.3344727
2: -5.1740036, 15.9146786, -5.3908873, 16.3171444, -21.4911461, 21.3055649
3: -5.7092237, 21.9910412, -5.9503188, 22.5346546, -28.2438774, 27.9413586
4: -4.5661373, 20.3150978, -4.7566547, 20.9033737, -25.4695110, 25.0717525

Time for backsubstitution: 2.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8510354, upper bound: 27.8503726
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8510354, upper bound: 27.8503726
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -3.5486126, 13.3772373, -3.9184031, 14.6225538, -18.1711655, 17.2956409
1: -5.9423294, 13.5300770, -6.5344067, 14.7736359, -20.7159653, 20.0644798
2: -4.7485104, 14.7520132, -5.2468700, 16.0727081, -20.8212185, 19.9988823
3: -5.2675343, 20.3012791, -5.7845664, 22.2014503, -27.4689846, 26.0858459
4: -4.2054482, 18.7917194, -4.6306186, 20.5310402, -24.7364883, 23.4223385

Time for backsubstitution: 2.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8503726, upper bound: 27.8510354
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8503726, upper bound: 27.8510354
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -3.9118266, 14.5160675, -3.9672580, 14.7736979, -18.6855221, 18.4833260
1: -6.5327997, 14.6775446, -6.6120090, 14.9280148, -21.4608154, 21.2895527
2: -5.2392712, 15.9570942, -5.3131099, 16.2373619, -21.4766312, 21.2702045
3: -5.7927489, 22.0449142, -5.8528652, 22.4279957, -28.2207432, 27.8977776
4: -4.6233401, 20.4146690, -4.6880665, 20.7485371, -25.3718777, 25.1027355

Time for backsubstitution: 2.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8503726, upper bound: 27.8510354
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8503726, upper bound: 27.8510354
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -3.5486126, 13.3772373, -3.9740572, 14.6925678, -18.2411785, 17.3512917
1: -5.9423294, 13.5300770, -6.6321397, 14.8614407, -20.8037682, 20.1622162
2: -4.7485104, 14.7520132, -5.3237147, 16.1516628, -20.9001732, 20.0757275
3: -5.2675343, 20.3012791, -5.8811302, 22.3070793, -27.5746136, 26.1824093
4: -4.2054482, 18.7917194, -4.6983848, 20.6851788, -24.8906269, 23.4901047

Time for backsubstitution: 2.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8518903, upper bound: 27.8518903
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8518903, upper bound: 27.8518903
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -3.9118266, 14.5160675, -4.0236449, 14.8443575, -18.7561798, 18.5397072
1: -6.5327997, 14.6775446, -6.7107162, 15.0167494, -21.5495472, 21.3882580
2: -5.2392712, 15.9570942, -5.3908873, 16.3171444, -21.5564137, 21.3479805
3: -5.7927489, 22.0449142, -5.9503188, 22.5346546, -28.3274021, 27.9952335
4: -4.6233401, 20.4146690, -4.7566547, 20.9033737, -25.5267124, 25.1713238

Time for backsubstitution: 2.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8518903, upper bound: 27.8518903
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8518903, upper bound: 27.8518903
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -3.9878480, 14.6482315, -5.0952649, 18.5291748, -22.5170231, 19.7434959
1: -6.6317897, 14.8119974, -8.4130306, 18.6891232, -25.3209133, 23.2250290
2: -5.3534884, 16.1117134, -6.8100896, 20.2712154, -25.6247025, 22.9218025
3: -5.8895788, 22.2077332, -7.4057221, 27.9912224, -33.8808022, 29.6134510
4: -4.7259350, 20.5818653, -5.9966183, 25.9766331, -30.7025681, 26.5784836

Time for backsubstitution: 2.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.4674971, upper bound: 27.4494078
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.5702566, upper bound: 27.5824524
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -3.8859746, 14.5205603, -5.1221609, 18.6184425, -22.5044174, 19.6427212
1: -6.4813662, 14.6674662, -8.4564943, 18.7807064, -25.2620716, 23.1239567
2: -5.2033811, 15.9605389, -6.8467531, 20.3681755, -25.5715561, 22.8072910
3: -5.7385736, 22.0517960, -7.4440994, 28.1251602, -33.8637352, 29.4958878
4: -4.5913091, 20.3842659, -6.0290127, 26.1052856, -30.6965923, 26.4132786

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7419981, upper bound: 27.7549919
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8246763, upper bound: 27.8294166
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -3.9420342, 14.5823555, -5.1221609, 18.6184425, -22.5604744, 19.7045155
1: -6.5784769, 14.7482300, -8.4564943, 18.7807064, -25.3591843, 23.2047234
2: -5.2801332, 16.0317326, -6.8467531, 20.3681755, -25.6483078, 22.8784828
3: -5.8338447, 22.1432133, -7.4440994, 28.1251602, -33.9589958, 29.5873127
4: -4.6589708, 20.5167599, -6.0290127, 26.1052856, -30.7642555, 26.5457726

Time for backsubstitution: 2.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7428530, upper bound: 27.7565096
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8219471, upper bound: 27.8291396
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -5.0158067, 18.3015442, -3.9672580, 14.7736979, -19.7895031, 22.2688026
1: -8.2864075, 18.4549599, -6.6120090, 14.9280148, -23.2144203, 25.0669670
2: -6.7030010, 20.0213966, -5.3131099, 16.2373619, -22.9403629, 25.3345070
3: -7.2928209, 27.6514435, -5.8528652, 22.4279957, -29.7208118, 33.5043068
4: -5.9033175, 25.6424065, -4.6880665, 20.7485371, -26.6518555, 30.3304691

Time for backsubstitution: 2.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8294166, upper bound: 27.8246763
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8294166, upper bound: 27.8246763
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -5.0158067, 18.3015442, -4.0236449, 14.8443575, -19.8601608, 22.3251877
1: -8.2864075, 18.4549599, -6.7107162, 15.0167494, -23.3031483, 25.1656723
2: -6.7030010, 20.0213966, -5.3908873, 16.3171444, -23.0201454, 25.4122849
3: -7.2928209, 27.6514435, -5.9503188, 22.5346546, -29.8274727, 33.6017609
4: -5.9033175, 25.6424065, -4.7566547, 20.9033737, -26.8066902, 30.3990612

Time for backsubstitution: 2.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8295960, upper bound: 27.8247772
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8295960, upper bound: 27.8247772
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -5.0045290, 18.2204800, -5.1221609, 18.6184425, -23.6229706, 23.3426399
1: -8.2631350, 18.3688087, -8.4564943, 18.7807064, -27.0438423, 26.8253021
2: -6.6878443, 19.9378281, -6.8467531, 20.3681755, -27.0560188, 26.7845802
3: -7.2701283, 27.5258446, -7.4440994, 28.1251602, -35.3952866, 34.9699402
4: -5.8837113, 25.5241604, -6.0290127, 26.1052856, -31.9889927, 31.5531731

Time for backsubstitution: 2.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7218969, upper bound: 27.7301504
time: 1.20 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8045752, upper bound: 27.8045752
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -3.3273158, 12.8372402, -5.2829285, 18.3765316, -21.7038479, 18.1201687
1: -5.6021156, 12.9728231, -8.7107792, 18.7208252, -24.3229408, 21.6835995
2: -4.4462528, 14.1458454, -7.0975385, 20.2531853, -24.6994381, 21.2433834
3: -4.9587526, 19.5276680, -7.7407331, 27.8119450, -32.7706985, 27.2684021
4: -3.9472556, 18.0581570, -6.2942061, 26.3107929, -30.2580490, 24.3523598

Time for backsubstitution: 2.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.5671579, upper bound: 27.5834311
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7692983, upper bound: 27.7522955
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8075718, upper bound: 27.7901619
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -3.8642793, 14.4800587, -5.3472810, 18.5768604, -22.4411392, 19.8273392
1: -6.4491496, 14.6237555, -8.8134174, 18.9273415, -25.3764915, 23.4371719
2: -5.1740036, 15.9146786, -7.1847754, 20.4710217, -25.6450233, 23.0994492
3: -5.7092237, 21.9910412, -7.8320279, 28.1110363, -33.8202591, 29.8230667
4: -4.5661373, 20.3150978, -6.3711629, 26.5981026, -31.1642380, 26.6862602

Time for backsubstitution: 2.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.3785061, upper bound: 27.4713643
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7678358, upper bound: 27.7522955
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8063275, upper bound: 27.7894296
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -3.3273158, 12.8372402, -5.2665782, 18.3989201, -21.7262363, 18.1038189
1: -5.6021156, 12.9728231, -8.7045412, 18.7308083, -24.3329239, 21.6773605
2: -4.4462528, 14.1458454, -7.0823565, 20.3064594, -24.7527122, 21.2282028
3: -4.9587526, 19.5276680, -7.7395706, 27.8700790, -32.8288307, 27.2672386
4: -3.9472556, 18.0581570, -6.2865882, 26.3321667, -30.2794189, 24.3447456

Time for backsubstitution: 2.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8489182, upper bound: 27.8427204
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8288838, upper bound: 27.8283290
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -3.8642793, 14.4800587, -5.3233452, 18.5752125, -22.4394913, 19.8034039
1: -6.4491496, 14.6237555, -8.7951794, 18.9134140, -25.3625641, 23.4189339
2: -5.1740036, 15.9146786, -7.1597281, 20.4994736, -25.6734772, 23.0744057
3: -5.7092237, 21.9910412, -7.8201966, 28.1347542, -33.8439789, 29.8112316
4: -4.5661373, 20.3150978, -6.3551607, 26.5873337, -31.1534634, 26.6702576

Time for backsubstitution: 2.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8476162, upper bound: 27.8414197
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8275818, upper bound: 27.8270283
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -3.5486126, 13.3772373, -5.2829285, 18.3765316, -21.9251423, 18.6601639
1: -5.9423294, 13.5300770, -8.7107792, 18.7208252, -24.6631527, 22.2408543
2: -4.7485104, 14.7520132, -7.0975385, 20.2531853, -25.0016956, 21.8495522
3: -5.2675343, 20.3012791, -7.7407331, 27.8119450, -33.0794716, 28.0420113
4: -4.2054482, 18.7917194, -6.2942061, 26.3107929, -30.5162411, 25.0859241

Time for backsubstitution: 2.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.5559922, upper bound: 27.5765199
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7314561, upper bound: 27.7111992
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8068519, upper bound: 27.7901787
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -3.9118266, 14.5160675, -5.3472810, 18.5768604, -22.4886875, 19.8633480
1: -6.5327997, 14.6775446, -8.8134174, 18.9273415, -25.4601402, 23.4909630
2: -5.2392712, 15.9570942, -7.1847754, 20.4710217, -25.7102928, 23.1418667
3: -5.7927489, 22.0449142, -7.8320279, 28.1110363, -33.9037857, 29.8769417
4: -4.6233401, 20.4146690, -6.3711629, 26.5981026, -31.2214432, 26.7858315

Time for backsubstitution: 2.61 seconds
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=30.45956039428711
rel_dist={0: [-27.85276297632284, 27.852762976322843]}

## Binary search (step 1) starts
Candidate diff: 0.1250000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8506507, upper bound: 27.8473542
time: 0.64 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8466200, upper bound: 27.8466200
time: 0.71 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.57 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.57
Output dim: 0, lower bound: -27.8506507, upper bound: 27.8473542
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.57
Output dim: 0, lower bound: -27.8466200, upper bound: 27.8466200

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -4.8150167, 17.2120781, -6.2440305, 21.3977070, -26.2127209, 23.4561081
1: -7.9723043, 17.4862099, -10.2540798, 21.8523140, -29.8246193, 27.7402897
2: -6.4574275, 18.9246101, -8.3976126, 23.5754948, -30.0329208, 27.3222237
3: -7.0645366, 26.1020927, -9.1058140, 32.3823471, -39.4468842, 35.2079048
4: -5.6939859, 24.4059410, -7.4231143, 30.6405563, -36.3345413, 31.8290520

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8151707, upper bound: 27.8102748
time: 0.67 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
time: 0.82 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -6.2154808, 21.2638779, -6.9769545, 23.4826069, -29.6980877, 28.2408295
1: -10.2115059, 21.7044182, -11.4056101, 24.0275745, -34.2390823, 33.1100273
2: -8.3589554, 23.4411602, -9.3722210, 25.8551235, -34.2140770, 32.8133774
3: -9.0649576, 32.1635170, -10.1196079, 35.4739799, -44.5389290, 42.2831268
4: -7.4015422, 30.4624748, -8.2898979, 33.6252289, -41.0267715, 38.7523727

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8466200, upper bound: 27.8466200
time: 0.56 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8466200, upper bound: 27.8466200
time: 0.66 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.90 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.90
Output dim: 0, lower bound: -27.8151707, upper bound: 27.8102748
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.90
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.90
Output dim: 0, lower bound: -27.8466200, upper bound: 27.8466200
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.90
Output dim: 0, lower bound: -27.8466200, upper bound: 27.8466200

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -4.8150167, 17.2120781, -5.4512777, 18.9793434, -23.7943573, 22.6633549
1: -7.9723043, 17.4862099, -8.9962053, 19.3352833, -27.3075867, 26.4824142
2: -6.4574275, 18.9246101, -7.3275404, 20.9384689, -27.3958969, 26.2521496
3: -7.0645366, 26.1020927, -7.9985380, 28.7568264, -35.8213615, 34.1006241
4: -5.6939859, 24.4059410, -6.4892559, 27.1696510, -32.8636360, 30.8951931

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
time: 0.94 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -4.5655317, 16.4501953, -6.2975054, 21.9635296, -26.5290604, 22.7476940
1: -7.5735898, 16.6911907, -10.3216124, 22.2796669, -29.8532562, 27.0128021
2: -6.1195216, 18.0840569, -8.4397888, 24.1041565, -30.2236748, 26.5238457
3: -6.7135277, 24.9447193, -9.1346750, 33.1087074, -39.8222313, 34.0793953
4: -5.3976588, 23.2858429, -7.4620428, 31.1666718, -36.5643272, 30.7478848

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
time: 0.67 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -6.2154808, 21.2638779, -4.8150167, 17.2120781, -23.4275589, 26.0788918
1: -10.2115059, 21.7044182, -7.9723043, 17.4862099, -27.6977158, 29.6767235
2: -8.3589554, 23.4411602, -6.4574275, 18.9246101, -27.2835655, 29.8985825
3: -9.0649576, 32.1635170, -7.0645366, 26.1020927, -35.1670456, 39.2280540
4: -7.4015422, 30.4624748, -5.6939859, 24.4059410, -31.8074837, 36.1564598

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7891958, upper bound: 27.7958566
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7601637, upper bound: 27.7601637
time: 0.76 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -6.2154808, 21.2638779, -6.2154808, 21.2638779, -27.4793587, 27.4793587
1: -10.2115059, 21.7044182, -10.2115059, 21.7044182, -31.9159241, 31.9159241
2: -8.3589554, 23.4411602, -8.3589554, 23.4411602, -31.8001156, 31.8001156
3: -9.0649576, 32.1635170, -9.0649576, 32.1635170, -41.2284698, 41.2284698
4: -7.4015422, 30.4624748, -7.4015422, 30.4624748, -37.8640175, 37.8640175

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7891958, upper bound: 27.7958566
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7601637, upper bound: 27.7601637
time: 0.68 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.20 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 0, lower bound: -27.7891958, upper bound: 27.7958566
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 4.20
Output dim: 0, lower bound: -27.7601637, upper bound: 27.7601637
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 0, lower bound: -27.7891958, upper bound: 27.7958566
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 4.20
Output dim: 0, lower bound: -27.7601637, upper bound: 27.7601637

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4.1239586, 15.1485586, -5.4512777, 18.9793434, -23.1033020, 20.5998325
1: -6.8710823, 15.3317356, -8.9962053, 19.3352833, -26.2063656, 24.3279419
2: -5.5255399, 16.6497955, -7.3275404, 20.9384689, -26.4640083, 23.9773331
3: -6.0924091, 22.9905643, -7.9985380, 28.7568264, -34.8492355, 30.9890957
4: -4.8740869, 21.3502178, -6.4892559, 27.1696510, -32.0437317, 27.8394718

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8151707, upper bound: 27.8102748
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8151707, upper bound: 27.8102748
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -5.2290773, 18.9489193, -5.4512777, 18.9793434, -24.2084179, 24.4001961
1: -8.6279545, 19.1228676, -8.9962053, 19.3352833, -27.9632378, 28.1190720
2: -6.9899049, 20.7296162, -7.3275404, 20.9384689, -27.9283733, 28.0571556
3: -7.5971055, 28.6222916, -7.9985380, 28.7568264, -36.3539314, 36.6208191
4: -6.1555519, 26.5859127, -6.4892559, 27.1696510, -33.3251953, 33.0751610

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8151707, upper bound: 27.8102748
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8151707, upper bound: 27.8102748
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4.1239586, 15.1485586, -6.2975054, 21.9635296, -26.0874863, 21.4460564
1: -6.8710823, 15.3317356, -10.3216124, 22.2796669, -29.1507492, 25.6533470
2: -5.5255399, 16.6497955, -8.4397888, 24.1041565, -29.6296959, 25.0895844
3: -6.0924091, 22.9905643, -9.1346750, 33.1087074, -39.2011185, 32.1252365
4: -4.8740869, 21.3502178, -7.4620428, 31.1666718, -36.0407524, 28.8122597

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -5.2290773, 18.9489193, -6.2975054, 21.9635296, -27.1926041, 25.2464218
1: -8.6279545, 19.1228676, -10.3216124, 22.2796669, -30.9076214, 29.4444790
2: -6.9899049, 20.7296162, -8.4397888, 24.1041565, -31.0940590, 29.1694050
3: -7.5971055, 28.6222916, -9.1346750, 33.1087074, -40.7058144, 37.7569580
4: -6.1555519, 26.5859127, -7.4620428, 31.1666718, -37.3222198, 34.0479546

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -5.4554358, 18.9849701, -4.8150167, 17.2120781, -22.6675148, 23.7999821
1: -9.0068264, 19.3360481, -7.9723043, 17.4862099, -26.4930363, 27.3083534
2: -7.3360786, 20.9468498, -6.4574275, 18.9246101, -26.2606888, 27.4042778
3: -8.0077810, 28.7492008, -7.0645366, 26.1020927, -34.1098709, 35.8137360
4: -6.5097780, 27.1807137, -5.6939859, 24.4059410, -30.9157124, 32.8746986

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7829412, upper bound: 27.7817977
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7829412, upper bound: 27.7817977
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -5.4554358, 18.9849701, -6.2154808, 21.2638779, -26.7193127, 25.2004509
1: -9.0068264, 19.3360481, -10.2115059, 21.7044182, -30.7112446, 29.5475540
2: -7.3360786, 20.9468498, -8.3589554, 23.4411602, -30.7772388, 29.3058052
3: -8.0077810, 28.7492008, -9.0649576, 32.1635170, -40.1712990, 37.8141594
4: -6.5097780, 27.1807137, -7.4015422, 30.4624748, -36.9722519, 34.5822563

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7601637, upper bound: 27.7601637
time: 0.96 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7601637, upper bound: 27.7601637
time: 0.73 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.46 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.46
Output dim: 0, lower bound: -27.8151707, upper bound: 27.8102748
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.46
Output dim: 0, lower bound: -27.8151707, upper bound: 27.8102748
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.46
Output dim: 0, lower bound: -27.8151707, upper bound: 27.8102748
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.46
Output dim: 0, lower bound: -27.8151707, upper bound: 27.8102748
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.46
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.46
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.46
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.46
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.46
Output dim: 0, lower bound: -27.7829412, upper bound: 27.7817977
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.46
Output dim: 0, lower bound: -27.7829412, upper bound: 27.7817977
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 4.46
Output dim: 0, lower bound: -27.7601637, upper bound: 27.7601637
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 4.46
Output dim: 0, lower bound: -27.7601637, upper bound: 27.7601637

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -4.1239586, 15.1485586, -4.1239586, 15.1485586, -19.2725143, 19.2725124
1: -6.8710823, 15.3317356, -6.8710823, 15.3317356, -22.2028179, 22.2028179
2: -5.5255399, 16.6497955, -5.5255399, 16.6497955, -22.1753349, 22.1753349
3: -6.0924091, 22.9905643, -6.0924091, 22.9905643, -29.0829735, 29.0829735
4: -4.8740869, 21.3502178, -4.8740869, 21.3502178, -26.2243042, 26.2243042

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8482950, upper bound: 27.8451396
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8506507, upper bound: 27.8473542
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -4.1239586, 15.1485586, -5.4495296, 18.9653072, -23.0892639, 20.5980816
1: -6.8710823, 15.3317356, -8.9974012, 19.3160076, -26.1870899, 24.3291359
2: -5.5255399, 16.6497955, -7.3282366, 20.9258022, -26.4513416, 23.9780312
3: -6.0924091, 22.9905643, -7.9993610, 28.7203865, -34.8127975, 30.9899139
4: -4.8740869, 21.3502178, -6.5029669, 27.1545448, -32.0286255, 27.8531818

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8482950, upper bound: 27.8451396
time: 1.41 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8506507, upper bound: 27.8473542
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -5.2290773, 18.9489193, -4.1239586, 15.1485586, -20.3776302, 23.0728779
1: -8.6279545, 19.1228676, -6.8710823, 15.3317356, -23.9596901, 25.9939499
2: -6.9899049, 20.7296162, -5.5255399, 16.6497955, -23.6396961, 26.2551556
3: -7.5971055, 28.6222916, -6.0924091, 22.9905643, -30.5876656, 34.7146988
4: -6.1555519, 26.5859127, -4.8740869, 21.3502178, -27.5057697, 31.4599972

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7332521, upper bound: 27.7214982
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8151707, upper bound: 27.8102748
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -5.2290773, 18.9489193, -5.4495296, 18.9653072, -24.1943798, 24.3984451
1: -8.6279545, 19.1228676, -8.9974012, 19.3160076, -27.9439621, 28.1202698
2: -6.9899049, 20.7296162, -7.3282366, 20.9258022, -27.9157028, 28.0578537
3: -7.5971055, 28.6222916, -7.9993610, 28.7203865, -36.3174934, 36.6216469
4: -6.1555519, 26.5859127, -6.5029669, 27.1545448, -33.3100891, 33.0888786

Time for backsubstitution: 2.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7332521, upper bound: 27.7214982
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8151707, upper bound: 27.8102748
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -4.1239586, 15.1485586, -5.3676920, 19.4284458, -23.5524025, 20.5162487
1: -6.8710823, 15.3317356, -8.8535709, 19.6024017, -26.4734840, 24.1853065
2: -5.5255399, 16.6497955, -7.1736655, 21.2490845, -26.7746239, 23.8234596
3: -6.0924091, 22.9905643, -7.7958689, 29.3602962, -35.4527054, 30.7864265
4: -4.8740869, 21.3502178, -6.3186235, 27.2952156, -32.1692963, 27.6688385

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7863139, upper bound: 27.7907424
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7934372, upper bound: 27.7965025
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4.1239586, 15.1485586, -6.3037281, 22.0140495, -26.1380081, 21.4522800
1: -6.8710823, 15.3317356, -10.3375835, 22.3265495, -29.1976318, 25.6693192
2: -5.5255399, 16.6497955, -8.4509039, 24.1487713, -29.6743107, 25.1006985
3: -6.0924091, 22.9905643, -9.1481276, 33.1661034, -39.2585144, 32.1386871
4: -4.8740869, 21.3502178, -7.4770923, 31.2366295, -36.1107101, 28.8273106

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7863139, upper bound: 27.7907424
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7934372, upper bound: 27.7965025
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -5.2290773, 18.9489193, -5.3676920, 19.4284458, -24.6575184, 24.3166122
1: -8.6279545, 19.1228676, -8.8535709, 19.6024017, -28.2303562, 27.9764366
2: -6.9899049, 20.7296162, -7.1736655, 21.2490845, -28.2389851, 27.9032822
3: -7.5971055, 28.6222916, -7.7958689, 29.3602962, -36.9574013, 36.4181519
4: -6.1555519, 26.5859127, -6.3186235, 27.2952156, -33.4507599, 32.9045372

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.6337764, upper bound: 27.6343352
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -5.2290773, 18.9489193, -6.3037281, 22.0140495, -27.2431240, 25.2526455
1: -8.6279545, 19.1228676, -10.3375835, 22.3265495, -30.9545040, 29.4604473
2: -6.9899049, 20.7296162, -8.4509039, 24.1487713, -31.1386719, 29.1805191
3: -7.5971055, 28.6222916, -9.1481276, 33.1661034, -40.7632065, 37.7704163
4: -6.1555519, 26.5859127, -7.4770923, 31.2366295, -37.3921814, 34.0630035

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.6337764, upper bound: 27.6343352
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -5.4554358, 18.9849701, -4.1239586, 15.1485586, -20.6039886, 23.1089268
1: -9.0068264, 19.3360481, -6.8710823, 15.3317356, -24.3385620, 26.2071304
2: -7.3360786, 20.9468498, -5.5255399, 16.6497955, -23.9858723, 26.4723892
3: -8.0077810, 28.7492008, -6.0924091, 22.9905643, -30.9983368, 34.8416100
4: -6.5097780, 27.1807137, -4.8740869, 21.3502178, -27.8599949, 32.0547943

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8102748, upper bound: 27.8151707
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8066735, upper bound: 27.8100959
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -5.4554358, 18.9849701, -5.2290773, 18.9489193, -24.4043541, 24.2140408
1: -9.0068264, 19.3360481, -8.6279545, 19.1228676, -28.1296940, 27.9640026
2: -7.3360786, 20.9468498, -6.9899049, 20.7296162, -28.0656948, 27.9367523
3: -8.0077810, 28.7492008, -7.5971055, 28.6222916, -36.6300621, 36.3463058
4: -6.5097780, 27.1807137, -6.1555519, 26.5859127, -33.0956841, 33.3362617

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8102748, upper bound: 27.8151707
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8066735, upper bound: 27.8100959
time: 0.62 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 6.31 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.31
Output dim: 0, lower bound: -27.8482950, upper bound: 27.8451396
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.31
Output dim: 0, lower bound: -27.8506507, upper bound: 27.8473542
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.31
Output dim: 0, lower bound: -27.8482950, upper bound: 27.8451396
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.31
Output dim: 0, lower bound: -27.8506507, upper bound: 27.8473542
IS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 6.31
Output dim: 0, lower bound: -27.7332521, upper bound: 27.7214982
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.31
Output dim: 0, lower bound: -27.8151707, upper bound: 27.8102748
IS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 6.31
Output dim: 0, lower bound: -27.7332521, upper bound: 27.7214982
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.31
Output dim: 0, lower bound: -27.8151707, upper bound: 27.8102748
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.31
Output dim: 0, lower bound: -27.7863139, upper bound: 27.7907424
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.31
Output dim: 0, lower bound: -27.7934372, upper bound: 27.7965025
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.31
Output dim: 0, lower bound: -27.7863139, upper bound: 27.7907424
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.31
Output dim: 0, lower bound: -27.7934372, upper bound: 27.7965025
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 6.31
Output dim: 0, lower bound: -27.6337764, upper bound: 27.6343352
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.31
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 6.31
Output dim: 0, lower bound: -27.6337764, upper bound: 27.6343352
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.31
Output dim: 0, lower bound: -27.7817977, upper bound: 27.7829412
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.31
Output dim: 0, lower bound: -27.8102748, upper bound: 27.8151707
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.31
Output dim: 0, lower bound: -27.8066735, upper bound: 27.8100959
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.31
Output dim: 0, lower bound: -27.8102748, upper bound: 27.8151707
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.31
Output dim: 0, lower bound: -27.8066735, upper bound: 27.8100959

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -3.9672580, 14.7736979, -4.0149913, 14.8063297, -18.7735882, 18.7886868
1: -6.6120090, 14.9280148, -6.6963530, 14.9833736, -21.5953751, 21.6243668
2: -5.3131099, 16.2373619, -5.3784075, 16.2774563, -21.5905666, 21.6157665
3: -5.8528652, 22.4279957, -5.9390106, 22.4791031, -28.3319645, 28.3670063
4: -4.6880665, 20.7485371, -4.7452264, 20.8504257, -25.5384903, 25.4937630

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8510354, upper bound: 27.8506168
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8510354, upper bound: 27.8503726
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -4.0236449, 14.8443575, -4.1239586, 15.1485586, -19.1721935, 18.9683132
1: -6.7107162, 15.0167494, -6.8710823, 15.3317356, -22.0424519, 21.8878307
2: -5.3908873, 16.3171444, -5.5255399, 16.6497955, -22.0406837, 21.8426838
3: -5.9503188, 22.5346546, -6.0924091, 22.9905643, -28.9408836, 28.6270638
4: -4.7566547, 20.9033737, -4.8740869, 21.3502178, -26.1068726, 25.7774601

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8521223, upper bound: 27.8521345
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8521223, upper bound: 27.8521345
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -3.9672580, 14.7736979, -5.3228903, 18.5630894, -22.5303478, 20.0965862
1: -6.6120090, 14.9280148, -8.7939138, 18.9027367, -25.5147457, 23.7219276
2: -5.3131099, 16.2373619, -7.1569538, 20.4865093, -25.7996197, 23.3943157
3: -5.8528652, 22.4279957, -7.8189101, 28.1192169, -33.9720840, 30.2468967
4: -4.6880665, 20.7485371, -6.3504553, 26.5697594, -31.2578239, 27.0989914

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7811249, upper bound: 27.7724519
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7811249, upper bound: 27.8451396
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -4.0236449, 14.8443575, -5.4495296, 18.9653072, -22.9889469, 20.2938824
1: -6.7107162, 15.0167494, -8.9974012, 19.3160076, -26.0267220, 24.0141506
2: -5.3908873, 16.3171444, -7.3282366, 20.9258022, -26.3166885, 23.6453819
3: -5.9503188, 22.5346546, -7.9993610, 28.7203865, -34.6707039, 30.5340118
4: -4.7566547, 20.9033737, -6.5029669, 27.1545448, -31.9111977, 27.4063396

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7811249, upper bound: 27.7724519
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7811249, upper bound: 27.8473542
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -5.1221609, 18.6184425, -4.1239586, 15.1485586, -20.2707195, 22.7424011
1: -8.4564943, 18.7807064, -6.8710823, 15.3317356, -23.7882309, 25.6517887
2: -6.8467531, 20.3681755, -5.5255399, 16.6497955, -23.4965458, 25.8937149
3: -7.4440994, 28.1251602, -6.0924091, 22.9905643, -30.4346581, 34.2175674
4: -6.0290127, 26.1052856, -4.8740869, 21.3502178, -27.3792305, 30.9793720

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8161993, upper bound: 27.8142713
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8161993, upper bound: 27.8166596
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -5.1221609, 18.6184425, -5.4495296, 18.9653072, -24.0874672, 24.0679684
1: -8.4564943, 18.7807064, -8.9974012, 19.3160076, -27.7725010, 27.7781067
2: -6.8467531, 20.3681755, -7.3282366, 20.9258022, -27.7725506, 27.6964111
3: -7.4440994, 28.1251602, -7.9993610, 28.7203865, -36.1644821, 36.1245155
4: -6.0290127, 26.1052856, -6.5029669, 27.1545448, -33.1835556, 32.6082535

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8151707, upper bound: 27.8102748
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8100959, upper bound: 27.8066735
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -3.9672580, 14.7736979, -5.2346349, 19.0006199, -22.9678783, 20.0083313
1: -6.6120090, 14.9280148, -8.6408272, 19.1678047, -25.7798138, 23.5688419
2: -5.3131099, 16.2373619, -6.9939442, 20.7856922, -26.0988026, 23.2313004
3: -5.8528652, 22.4279957, -7.6067905, 28.7209797, -34.5738449, 30.0347824
4: -4.6880665, 20.7485371, -6.1610508, 26.6810074, -31.3690739, 26.9095879

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7369988, upper bound: 27.7491186
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7369988, upper bound: 27.8161993
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -4.0236449, 14.8443575, -5.3676920, 19.4284458, -23.4520855, 20.2120476
1: -6.7107162, 15.0167494, -8.8535709, 19.6024017, -26.3131180, 23.8703194
2: -5.3908873, 16.3171444, -7.1736655, 21.2490845, -26.6399727, 23.4908104
3: -5.9503188, 22.5346546, -7.7958689, 29.3602962, -35.3106155, 30.3305206
4: -4.7566547, 20.9033737, -6.3186235, 27.2952156, -32.0518684, 27.2219963

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7369988, upper bound: 27.7491308
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7369988, upper bound: 27.8186464
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -3.9672580, 14.7736979, -6.1708078, 21.5981865, -25.5654449, 20.9445038
1: -6.6120090, 14.9280148, -10.1251163, 21.8951111, -28.5071201, 25.0531311
2: -5.3131099, 16.2373619, -8.2717133, 23.6972084, -29.0103168, 24.5090714
3: -5.8528652, 22.4279957, -8.9592447, 32.5481529, -38.4010162, 31.3872375
4: -4.6880665, 20.7485371, -7.3196201, 30.6385307, -35.3265991, 28.0681572

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.3827311, upper bound: 27.4464384
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.1831300, upper bound: 27.0920167
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -4.0236449, 14.8443575, -6.3037281, 22.0140495, -26.0376911, 21.1480808
1: -6.7107162, 15.0167494, -10.3375835, 22.3265495, -29.0372658, 25.3543320
2: -5.3908873, 16.3171444, -8.4509039, 24.1487713, -29.5396576, 24.7680473
3: -5.9503188, 22.5346546, -9.1481276, 33.1661034, -39.1164207, 31.6827812
4: -4.7566547, 20.9033737, -7.4770923, 31.2366295, -35.9932785, 28.3804665

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.4136383, upper bound: 27.4796633
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.1801368, upper bound: 27.0923805
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -5.1221609, 18.6184425, -5.3676920, 19.4284458, -24.5506058, 23.9861336
1: -8.4564943, 18.7807064, -8.8535709, 19.6024017, -28.0588951, 27.6342754
2: -6.8467531, 20.3681755, -7.1736655, 21.2490845, -28.0958347, 27.5418415
3: -7.4440994, 28.1251602, -7.7958689, 29.3602962, -36.8043938, 35.9210281
4: -6.0290127, 26.1052856, -6.3186235, 27.2952156, -33.3242264, 32.4239082

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.6887317, upper bound: 27.6921718
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.6887317, upper bound: 27.8045752
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -5.1221609, 18.6184425, -6.3037281, 22.0140495, -27.1362114, 24.9221687
1: -8.4564943, 18.7807064, -10.3375835, 22.3265495, -30.7830429, 29.1182899
2: -6.8467531, 20.3681755, -8.4509039, 24.1487713, -30.9955254, 28.8190804
3: -7.4440994, 28.1251602, -9.1481276, 33.1661034, -40.6102028, 37.2732849
4: -6.0290127, 26.1052856, -7.4770923, 31.2366295, -37.2656403, 33.5823784

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.4255380, upper bound: 27.4872790
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.1875214, upper bound: 27.0961534
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -5.3486147, 18.6461754, -4.1239586, 15.1485586, -20.4971714, 22.7701321
1: -8.8365345, 18.9885597, -6.8710823, 15.3317356, -24.1682701, 25.8596420
2: -7.1932240, 20.5773335, -5.5255399, 16.6497955, -23.8430195, 26.1028728
3: -7.8570271, 28.2433796, -6.0924091, 22.9905643, -30.8475876, 34.3357887
4: -6.3842969, 26.6930981, -4.8740869, 21.3502178, -27.7345104, 31.5671844

Time for backsubstitution: 2.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8066490, upper bound: 27.8087040
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8319303, upper bound: 27.8338886
time: 0.95 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8307342, upper bound: 27.8333720
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -6.2936206, 21.7869453, -4.0524225, 14.9171429, -21.2107620, 25.8393669
1: -10.3439865, 22.1925621, -6.7571959, 15.1003246, -25.4443111, 28.9497585
2: -8.4501743, 23.9575081, -5.4290342, 16.4011402, -24.8513145, 29.3865395
3: -9.1837044, 32.9127502, -5.9927616, 22.6568508, -31.8405552, 38.9055099
4: -7.4634824, 31.1625271, -4.7899675, 21.0294399, -28.4929218, 35.9524918

Time for backsubstitution: 2.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8246465, upper bound: 27.8253868
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8246888, upper bound: 27.8250651
time: 1.10 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -5.3486147, 18.6461754, -5.2290773, 18.9489193, -24.2975349, 23.8752480
1: -8.8365345, 18.9885597, -8.6279545, 19.1228676, -27.9594021, 27.6165142
2: -7.1932240, 20.5773335, -6.9899049, 20.7296162, -27.9228401, 27.5672359
3: -7.8570271, 28.2433796, -7.5971055, 28.6222916, -36.4793167, 35.8404846
4: -6.3842969, 26.6930981, -6.1555519, 26.5859127, -32.9702072, 32.8486481

Time for backsubstitution: 2.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7605614, upper bound: 27.7708995
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7530056, upper bound: 27.7609829
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7530056, upper bound: 27.7609829
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -6.2936206, 21.7869453, -5.1291890, 18.6285992, -24.9222183, 26.9161320
1: -10.3439865, 22.1925621, -8.4724665, 18.8008251, -29.1448116, 30.6650238
2: -8.4501743, 23.9575081, -6.8560882, 20.3877754, -28.8379440, 30.8135948
3: -9.1837044, 32.9127502, -7.4600296, 28.1665897, -37.3502960, 40.3727798
4: -7.4634824, 31.1625271, -6.0412002, 26.1446724, -33.6081543, 37.2037163

Time for backsubstitution: 2.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7530056, upper bound: 27.7609829
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7530056, upper bound: 27.8100959
time: 0.88 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 6.73 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -27.8510354, upper bound: 27.8506168
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -27.8510354, upper bound: 27.8503726
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -27.8521223, upper bound: 27.8521345
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -27.8521223, upper bound: 27.8521345
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -27.7811249, upper bound: 27.7724519
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -27.7811249, upper bound: 27.8451396
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -27.7811249, upper bound: 27.7724519
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -27.7811249, upper bound: 27.8473542
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -27.8161993, upper bound: 27.8142713
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -27.8161993, upper bound: 27.8166596
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -27.8151707, upper bound: 27.8102748
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -27.8100959, upper bound: 27.8066735
IS_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 6.73
Output dim: 0, lower bound: -27.7369988, upper bound: 27.7491186
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -27.7369988, upper bound: 27.8161993
IS_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 6.73
Output dim: 0, lower bound: -27.7369988, upper bound: 27.7491308
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -27.7369988, upper bound: 27.8186464
IS_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 6.73
Output dim: 0, lower bound: -27.3827311, upper bound: 27.4464384
IS_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 6.73
Output dim: 0, lower bound: -27.1831300, upper bound: 27.0920167
IS_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 6.73
Output dim: 0, lower bound: -27.4136383, upper bound: 27.4796633
IS_A1_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 6.73
Output dim: 0, lower bound: -27.1801368, upper bound: 27.0923805
IS_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 6.73
Output dim: 0, lower bound: -27.6887317, upper bound: 27.6921718
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -27.6887317, upper bound: 27.8045752
IS_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 6.73
Output dim: 0, lower bound: -27.4255380, upper bound: 27.4872790
IS_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 6.73
Output dim: 0, lower bound: -27.1875214, upper bound: 27.0961534
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -27.8319303, upper bound: 27.8338886
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -27.8307342, upper bound: 27.8333720
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -27.8246465, upper bound: 27.8253868
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -27.8246888, upper bound: 27.8250651
IS_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 6.73
Output dim: 0, lower bound: -27.7530056, upper bound: 27.7609829
IS_A2_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 6.73
Output dim: 0, lower bound: -27.7530056, upper bound: 27.7609829
IS_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 6.73
Output dim: 0, lower bound: -27.7530056, upper bound: 27.7609829
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.73
Output dim: 0, lower bound: -27.7530056, upper bound: 27.8100959

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -3.7463884, 14.0923586, -3.7146320, 13.9655752, -17.7119637, 17.8069916
1: -6.2613506, 14.2316761, -6.2315459, 14.1693745, -20.4307251, 20.4632206
2: -5.0135727, 15.4947720, -4.9727449, 15.3927155, -20.4062881, 20.4675159
3: -5.5441704, 21.4066582, -5.5461183, 21.2158108, -26.7599812, 26.9527760
4: -4.4283752, 19.7668190, -4.4445453, 19.7290211, -24.1573963, 24.2113647

Time for backsubstitution: 2.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8510354, upper bound: 27.8503726
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8510354, upper bound: 27.8503726
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -3.9672580, 14.7736979, -3.9025440, 14.4742680, -18.4415264, 18.6762428
1: -6.6120090, 14.9280148, -6.5171070, 14.6402454, -21.2522545, 21.4451218
2: -5.3131099, 16.2373619, -5.2257447, 15.9133148, -21.2264233, 21.4631023
3: -5.8528652, 22.4279957, -5.7802448, 21.9835300, -27.8363934, 28.2082405
4: -4.6880665, 20.7485371, -4.6109204, 20.3557930, -25.0438595, 25.3594570

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8510354, upper bound: 27.8503726
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8510354, upper bound: 27.8503726
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -4.0236449, 14.8443575, -3.9672580, 14.7736979, -18.7973366, 18.8116150
1: -6.7107162, 15.0167494, -6.6120090, 14.9280148, -21.6387310, 21.6287575
2: -5.3908873, 16.3171444, -5.3131099, 16.2373619, -21.6282501, 21.6302547
3: -5.9503188, 22.5346546, -5.8528652, 22.4279957, -28.3783150, 28.3875179
4: -4.7566547, 20.9033737, -4.6880665, 20.7485371, -25.5051918, 25.5914402

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8506168, upper bound: 27.8510354
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8503726, upper bound: 27.8510354
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -4.0236449, 14.8443575, -4.0236449, 14.8443575, -18.8679943, 18.8679943
1: -6.7107162, 15.0167494, -6.7107162, 15.0167494, -21.7274647, 21.7274628
2: -5.3908873, 16.3171444, -5.3908873, 16.3171444, -21.7080307, 21.7080307
3: -5.9503188, 22.5346546, -5.9503188, 22.5346546, -28.4849739, 28.4849739
4: -4.7566547, 20.9033737, -4.7566547, 20.9033737, -25.6600285, 25.6600285

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8506168, upper bound: 27.8518903
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8503726, upper bound: 27.8518903
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -3.9672580, 14.7736979, -5.3342867, 18.5256844, -22.4929428, 20.1079845
1: -6.6120090, 14.9280148, -8.7927580, 18.8779011, -25.4899101, 23.7207718
2: -5.3131099, 16.2373619, -7.1675220, 20.4197502, -25.7328606, 23.4048824
3: -5.8528652, 22.4279957, -7.8134527, 28.0398369, -33.8927002, 30.2414436
4: -4.6880665, 20.7485371, -6.3561907, 26.5359650, -31.2240314, 27.1047287

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7832727, upper bound: 27.7724734
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7819283, upper bound: 27.7724734
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -3.9672580, 14.7736979, -5.3180194, 18.5575352, -22.5247917, 20.0917149
1: -6.6120090, 14.9280148, -8.7866850, 18.8954010, -25.5074081, 23.7146988
2: -5.3131099, 16.2373619, -7.1526365, 20.4805336, -25.7936440, 23.3899918
3: -5.8528652, 22.4279957, -7.8126078, 28.1088581, -33.9617233, 30.2405949
4: -4.6880665, 20.7485371, -6.3490210, 26.5637913, -31.2518559, 27.0975571

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7832727, upper bound: 27.8451396
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7819283, upper bound: 27.8424193
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4.0236449, 14.8443575, -5.3342867, 18.5256844, -22.5493240, 20.1786442
1: -6.7107162, 15.0167494, -8.7927580, 18.8779011, -25.5886154, 23.8095055
2: -5.3908873, 16.3171444, -7.1675220, 20.4197502, -25.8106384, 23.4846668
3: -5.9503188, 22.5346546, -7.8134527, 28.0398369, -33.9901543, 30.3481026
4: -4.7566547, 20.9033737, -6.3561907, 26.5359650, -31.2926197, 27.2595634

Time for backsubstitution: 2.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7373582, upper bound: 27.7214762
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7811249, upper bound: 27.7724519
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4.0236449, 14.8443575, -5.3180194, 18.5575352, -22.5811729, 20.1623745
1: -6.7107162, 15.0167494, -8.7866850, 18.8954010, -25.6061134, 23.8034344
2: -5.3908873, 16.3171444, -7.1526365, 20.4805336, -25.8714218, 23.4697762
3: -5.9503188, 22.5346546, -7.8126078, 28.1088581, -34.0591736, 30.3472576
4: -4.7566547, 20.9033737, -6.3490210, 26.5637913, -31.3204460, 27.2523956

Time for backsubstitution: 2.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7373582, upper bound: 27.7214762
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7811249, upper bound: 27.8466200
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -5.1221609, 18.6184425, -3.9672580, 14.7736979, -19.8958588, 22.5857010
1: -8.4564943, 18.7807064, -6.6120090, 14.9280148, -23.3845100, 25.3927155
2: -6.8467531, 20.3681755, -5.3131099, 16.2373619, -23.0841103, 25.6812859
3: -7.4440994, 28.1251602, -5.8528652, 22.4279957, -29.8720913, 33.9780273
4: -6.0290127, 26.1052856, -4.6880665, 20.7485371, -26.7775497, 30.7933521

Time for backsubstitution: 2.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7549919, upper bound: 27.7419981
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8161993, upper bound: 27.8142713
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -5.1221609, 18.6184425, -4.0236449, 14.8443575, -19.9665184, 22.6420822
1: -8.4564943, 18.7807064, -6.7107162, 15.0167494, -23.4732437, 25.4914207
2: -6.8467531, 20.3681755, -5.3908873, 16.3171444, -23.1638947, 25.7590637
3: -7.4440994, 28.1251602, -5.9503188, 22.5346546, -29.9787521, 34.0754776
4: -6.0290127, 26.1052856, -4.7566547, 20.9033737, -26.9323864, 30.8619404

Time for backsubstitution: 2.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7549919, upper bound: 27.7423736
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8161993, upper bound: 27.8166596
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -5.1221609, 18.6184425, -5.3430967, 18.6277962, -23.7499580, 23.9615402
1: -8.4564943, 18.7807064, -8.8277473, 18.9698372, -27.4263306, 27.6084538
2: -6.8467531, 20.3681755, -7.1858878, 20.5576668, -27.4044189, 27.5540619
3: -7.4440994, 28.1251602, -7.8491683, 28.2164726, -35.6605682, 35.9743271
4: -6.0290127, 26.1052856, -6.3779392, 26.6686516, -32.6976624, 32.4832230

Time for backsubstitution: 2.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7708995, upper bound: 27.7605614
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7609829, upper bound: 27.7530056
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7609829, upper bound: 27.7530056
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -5.0218372, 18.2959957, -6.2870283, 21.7650471, -26.7868805, 24.5830231
1: -8.2999973, 18.4565258, -10.3334351, 22.1699352, -30.4699306, 28.7899590
2: -6.7120724, 20.0241451, -8.4414139, 23.9337292, -30.6458015, 28.4655571
3: -7.3060946, 27.6665726, -9.1742249, 32.8804474, -40.1865349, 36.8407936
4: -5.9139519, 25.6611958, -7.4558358, 31.1329460, -37.0468979, 33.1170235

Time for backsubstitution: 2.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7626733, upper bound: 27.7522794
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7626733, upper bound: 27.8061516
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -3.9672580, 14.7736979, -5.2570248, 19.0853691, -23.0526276, 20.0307198
1: -6.6120090, 14.9280148, -8.6756134, 19.2474518, -25.8594608, 23.6036282
2: -5.3131099, 16.2373619, -7.0249491, 20.8739109, -26.1870213, 23.2623081
3: -5.8528652, 22.4279957, -7.6366882, 28.8440228, -34.6968842, 30.0646763
4: -4.6880665, 20.7485371, -6.1873622, 26.7960167, -31.4840832, 26.9358997

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7187981, upper bound: 27.7633343
time: 1.10 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7376273, upper bound: 27.8161993
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -4.0236449, 14.8443575, -5.2570248, 19.0853691, -23.1090088, 20.1013775
1: -6.7107162, 15.0167494, -8.6756134, 19.2474518, -25.9581661, 23.6923637
2: -5.3908873, 16.3171444, -7.0249491, 20.8739109, -26.2647972, 23.3420887
3: -5.9503188, 22.5346546, -7.6366882, 28.8440228, -34.7943344, 30.1713390
4: -4.7566547, 20.9033737, -6.1873622, 26.7960167, -31.5526714, 27.0907364

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.4499634, upper bound: 27.5168724
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7369988, upper bound: 27.8183679
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -5.1221609, 18.6184425, -5.2570248, 19.0853691, -24.2075310, 23.8754654
1: -8.4564943, 18.7807064, -8.6756134, 19.2474518, -27.7039452, 27.4563198
2: -6.8467531, 20.3681755, -7.0249491, 20.8739109, -27.7206631, 27.3931198
3: -7.4440994, 28.1251602, -7.6366882, 28.8440228, -36.2881088, 35.7618484
4: -6.0290127, 26.1052856, -6.1873622, 26.7960167, -32.8250275, 32.2926483

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.4128059, upper bound: 27.4901681
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.6887317, upper bound: 27.8045751
time: 1.04 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -5.2237077, 18.2489967, -3.9672580, 14.7736979, -19.9974041, 22.2162533
1: -8.6356497, 18.5810204, -6.6120090, 14.9280148, -23.5636635, 25.1930294
2: -7.0239763, 20.1432285, -5.3131099, 16.2373619, -23.2613335, 25.4563389
3: -7.6788440, 27.6488400, -5.8528652, 22.4279957, -30.1068401, 33.5017014
4: -6.2335467, 26.1152992, -4.6880665, 20.7485371, -26.9820843, 30.8033638

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8310524, upper bound: 27.8329924
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 27
type: A, layer: 3, pos: 46
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 1

Time for candidate selection: 6.25 seconds

### Candidate
type: A, layer: 3, pos: 12

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 31

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 47

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 13

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8318237, upper bound: 27.8333779
time: 1.04 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8319303, upper bound: 27.8338886
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -5.3486147, 18.6461754, -4.0236449, 14.8443575, -20.1929703, 22.6698170
1: -8.8365345, 18.9885597, -6.7107162, 15.0167494, -23.8532829, 25.6992741
2: -7.1932240, 20.5773335, -5.3908873, 16.3171444, -23.5103683, 25.9682198
3: -7.8570271, 28.2433796, -5.9503188, 22.5346546, -30.3916817, 34.1936989
4: -6.3842969, 26.6930981, -4.7566547, 20.9033737, -27.2876701, 31.4497528

Time for backsubstitution: 2.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7463151, upper bound: 27.7587267
time: 0.95 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8265325, upper bound: 27.8301580
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8212345, upper bound: 27.8246741
time: 0.97 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6.1707392, 21.4091759, -3.8977401, 14.5425301, -20.7132645, 25.3069153
1: -10.1469593, 21.7977848, -6.5010967, 14.6984377, -24.8453903, 28.2988777
2: -8.2846413, 23.5388832, -5.2193527, 15.9899063, -24.2745457, 28.7582359
3: -9.0088835, 32.3427086, -5.7560048, 22.0955276, -31.1044121, 38.0987129
4: -7.3162203, 30.6088409, -4.6067400, 20.4317398, -27.7479591, 35.2155800

Time for backsubstitution: 3.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 46
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 27
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 2

Time for candidate selection: 6.20 seconds

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 12

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 31

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 47

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 46

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8246465, upper bound: 27.8253766
time: 1.04 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8235890, upper bound: 27.8253868
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6.2936206, 21.7869453, -3.9541659, 14.6180716, -20.9116879, 25.7411098
1: -10.3439865, 22.1925621, -6.5999050, 14.7914343, -25.1354179, 28.7924671
2: -8.4501743, 23.9575081, -5.2969570, 16.0740280, -24.5242004, 29.2544651
3: -9.1837044, 32.9127502, -5.8535643, 22.2083130, -31.3920174, 38.7663155
4: -7.4634824, 31.1625271, -4.6749430, 20.5903416, -28.0538235, 35.8374672

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 46
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 27
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 2

Time for candidate selection: 5.07 seconds

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 12

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 31

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 47

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 46

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8246888, upper bound: 27.8250553
time: 1.04 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8236313, upper bound: 27.8250651
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6.2936206, 21.7869453, -5.9018812, 21.2415199, -27.5351391, 27.6888275
1: -10.3439865, 22.1925621, -9.7087307, 21.4427299, -31.7867107, 31.9012928
2: -8.4501743, 23.9575081, -7.8944397, 23.2121582, -31.6623325, 31.8519478
3: -9.1837044, 32.9127502, -8.5615578, 32.1011086, -41.2848091, 41.4743042
4: -7.4634824, 31.1625271, -6.9596791, 29.8811989, -37.3446770, 38.1222038

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 46
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 27
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 2

Time for candidate selection: 5.37 seconds

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 12

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 31

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 47

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 46

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 13

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.6705713, upper bound: 27.6732548
time: 1.36 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7467127, upper bound: 27.7805032
time: 1.07 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7485267, upper bound: 27.7805578
time: 1.07 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 16.79 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 16.79
Output dim: 0, lower bound: -27.8510354, upper bound: 27.8503726
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 16.79
Output dim: 0, lower bound: -27.8510354, upper bound: 27.8503726
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 16.79
Output dim: 0, lower bound: -27.8510354, upper bound: 27.8503726
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 16.79
Output dim: 0, lower bound: -27.8510354, upper bound: 27.8503726
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 16.79
Output dim: 0, lower bound: -27.8506168, upper bound: 27.8510354
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 16.79
Output dim: 0, lower bound: -27.8503726, upper bound: 27.8510354
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 16.79
Output dim: 0, lower bound: -27.8506168, upper bound: 27.8518903
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 16.79
Output dim: 0, lower bound: -27.8503726, upper bound: 27.8518903
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 16.79
Output dim: 0, lower bound: -27.7832727, upper bound: 27.7724734
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 16.79
Output dim: 0, lower bound: -27.7819283, upper bound: 27.7724734
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 16.79
Output dim: 0, lower bound: -27.7832727, upper bound: 27.8451396
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 16.79
Output dim: 0, lower bound: -27.7819283, upper bound: 27.8424193
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 16.79
Output dim: 0, lower bound: -27.7373582, upper bound: 27.7214762
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 16.79
Output dim: 0, lower bound: -27.7811249, upper bound: 27.7724519
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 16.79
Output dim: 0, lower bound: -27.7373582, upper bound: 27.7214762
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 16.79
Output dim: 0, lower bound: -27.7811249, upper bound: 27.8466200
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 16.79
Output dim: 0, lower bound: -27.7549919, upper bound: 27.7419981
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 16.79
Output dim: 0, lower bound: -27.8161993, upper bound: 27.8142713
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 16.79
Output dim: 0, lower bound: -27.7549919, upper bound: 27.7423736
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 16.79
Output dim: 0, lower bound: -27.8161993, upper bound: 27.8166596
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 16.79
Output dim: 0, lower bound: -27.7609829, upper bound: 27.7530056
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 16.79
Output dim: 0, lower bound: -27.7609829, upper bound: 27.7530056
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 16.79
Output dim: 0, lower bound: -27.7626733, upper bound: 27.7522794
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 16.79
Output dim: 0, lower bound: -27.7626733, upper bound: 27.8061516
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 16.79
Output dim: 0, lower bound: -27.7187981, upper bound: 27.7633343
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 16.79
Output dim: 0, lower bound: -27.7376273, upper bound: 27.8161993
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 16.79
Output dim: 0, lower bound: -27.4499634, upper bound: 27.5168724
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 16.79
Output dim: 0, lower bound: -27.7369988, upper bound: 27.8183679
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 16.79
Output dim: 0, lower bound: -27.4128059, upper bound: 27.4901681
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 16.79
Output dim: 0, lower bound: -27.6887317, upper bound: 27.8045751
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 16.79
Output dim: 0, lower bound: -27.8318237, upper bound: 27.8333779
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 16.79
Output dim: 0, lower bound: -27.8319303, upper bound: 27.8338886
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 16.79
Output dim: 0, lower bound: -27.8265325, upper bound: 27.8301580
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 16.79
Output dim: 0, lower bound: -27.8212345, upper bound: 27.8246741
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 16.79
Output dim: 0, lower bound: -27.8246465, upper bound: 27.8253766
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 16.79
Output dim: 0, lower bound: -27.8235890, upper bound: 27.8253868
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 16.79
Output dim: 0, lower bound: -27.8246888, upper bound: 27.8250553
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 16.79
Output dim: 0, lower bound: -27.8236313, upper bound: 27.8250651
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 16.79
Output dim: 0, lower bound: -27.7467127, upper bound: 27.7805032
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 16.79
Output dim: 0, lower bound: -27.7485267, upper bound: 27.7805578

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -3.3273158, 12.8372402, -3.7146320, 13.9655752, -17.2928905, 16.5518703
1: -5.6021156, 12.9728231, -6.2315459, 14.1693745, -19.7714901, 19.2043686
2: -4.4462528, 14.1458454, -4.9727449, 15.3927155, -19.8389664, 19.1185875
3: -4.9587526, 19.5276680, -5.5461183, 21.2158108, -26.1745644, 25.0737858
4: -3.9472556, 18.0581570, -4.4445453, 19.7290211, -23.6762772, 22.5027008

Time for backsubstitution: 2.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8510233, upper bound: 27.8506168
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8510233, upper bound: 27.8506168
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -3.8642793, 14.4800587, -3.7146320, 13.9655752, -17.8298550, 18.1946907
1: -6.4491496, 14.6237555, -6.2315459, 14.1693745, -20.6185246, 20.8553009
2: -5.1740036, 15.9146786, -4.9727449, 15.3927155, -20.5667152, 20.8874207
3: -5.7092237, 21.9910412, -5.5461183, 21.2158108, -26.9250336, 27.5371590
4: -4.5661373, 20.3150978, -4.4445453, 19.7290211, -24.2951584, 24.7596436

Time for backsubstitution: 2.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8510233, upper bound: 27.8506168
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8510233, upper bound: 27.8506168
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -3.3273158, 12.8372402, -3.9025440, 14.4742680, -17.8015842, 16.7397842
1: -5.6021156, 12.9728231, -6.5171070, 14.6402454, -20.2423611, 19.4899273
2: -4.4462528, 14.1458454, -5.2257447, 15.9133148, -20.3595676, 19.3715897
3: -4.9587526, 19.5276680, -5.7802448, 21.9835300, -26.9422836, 25.3079128
4: -3.9472556, 18.0581570, -4.6109204, 20.3557930, -24.3030491, 22.6690769

Time for backsubstitution: 2.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8495177, upper bound: 27.8495177
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8495177, upper bound: 27.8503726
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -3.8642793, 14.4800587, -3.9025440, 14.4742680, -18.3385468, 18.3826027
1: -6.4491496, 14.6237555, -6.5171070, 14.6402454, -21.0893955, 21.1408615
2: -5.1740036, 15.9146786, -5.2257447, 15.9133148, -21.0873146, 21.1404209
3: -5.7092237, 21.9910412, -5.7802448, 21.9835300, -27.6927528, 27.7712860
4: -4.5661373, 20.3150978, -4.6109204, 20.3557930, -24.9219303, 24.9260178

Time for backsubstitution: 2.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8495177, upper bound: 27.8495177
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8495177, upper bound: 27.8503726
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -3.5486126, 13.3772373, -3.7463884, 14.0923586, -17.6409721, 17.1236229
1: -5.9423294, 13.5300770, -6.2613506, 14.2316761, -20.1740017, 19.7914276
2: -4.7485104, 14.7520132, -5.0135727, 15.4947720, -20.2432823, 19.7655869
3: -5.2675343, 20.3012791, -5.5441704, 21.4066582, -26.6741924, 25.8454494
4: -4.2054482, 18.7917194, -4.4283752, 19.7668190, -23.9722672, 23.2200947

Time for backsubstitution: 2.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8503726, upper bound: 27.8510354
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8503726, upper bound: 27.8510354
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -3.9118266, 14.5160675, -3.9672580, 14.7736979, -18.6855221, 18.4833260
1: -6.5327997, 14.6775446, -6.6120090, 14.9280148, -21.4608154, 21.2895527
2: -5.2392712, 15.9570942, -5.3131099, 16.2373619, -21.4766312, 21.2702045
3: -5.7927489, 22.0449142, -5.8528652, 22.4279957, -28.2207432, 27.8977776
4: -4.6233401, 20.4146690, -4.6880665, 20.7485371, -25.3718777, 25.1027355

Time for backsubstitution: 2.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8503726, upper bound: 27.8510354
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8503726, upper bound: 27.8510354
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -3.5486126, 13.3772373, -3.8009667, 14.1587753, -17.7073879, 17.1782036
1: -5.9423294, 13.5300770, -6.3575850, 14.3150053, -20.2573280, 19.8876553
2: -4.7485104, 14.7520132, -5.0889645, 15.5696754, -20.3181858, 19.8409748
3: -5.2675343, 20.3012791, -5.6391125, 21.5069199, -26.7744541, 25.9403896
4: -4.2054482, 18.7917194, -4.4945235, 19.9177952, -24.1232433, 23.2862434

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8495177, upper bound: 27.8518903
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8518903, upper bound: 27.8518903
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -3.9118266, 14.5160675, -4.0236449, 14.8443575, -18.7561798, 18.5397072
1: -6.5327997, 14.6775446, -6.7107162, 15.0167494, -21.5495472, 21.3882580
2: -5.2392712, 15.9570942, -5.3908873, 16.3171444, -21.5564137, 21.3479805
3: -5.7927489, 22.0449142, -5.9503188, 22.5346546, -28.3274021, 27.9952335
4: -4.6233401, 20.4146690, -4.7566547, 20.9033737, -25.5267124, 25.1713238

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8518903, upper bound: 27.8518903
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8518903, upper bound: 27.8518903
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -3.3273158, 12.8372402, -5.0554085, 17.6688290, -20.9961452, 17.8926487
1: -5.6021156, 12.9728231, -8.3481455, 17.9913807, -23.5934963, 21.3209667
2: -4.4462528, 14.1458454, -6.7890682, 19.4844379, -23.9306908, 20.9349136
3: -4.9587526, 19.5276680, -7.4178882, 26.7583866, -31.7171383, 26.9455547
4: -3.9472556, 18.0581570, -6.0218768, 25.2970486, -29.2443047, 24.0800304

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.4642401, upper bound: 27.5096569
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7413821, upper bound: 27.7330776
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7832727, upper bound: 27.7724734
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -3.8642793, 14.4800587, -5.3342867, 18.5256844, -22.3899632, 19.8143463
1: -6.4491496, 14.6237555, -8.7927580, 18.8779011, -25.3270512, 23.4165134
2: -5.1740036, 15.9146786, -7.1675220, 20.4197502, -25.5937538, 23.0822010
3: -5.7092237, 21.9910412, -7.8134527, 28.0398369, -33.7490616, 29.8044930
4: -4.5661373, 20.3150978, -6.3561907, 26.5359650, -31.1021023, 26.6712875

Time for backsubstitution: 2.50 seconds
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=30.45956039428711
rel_dist={0: [-27.85276297632284, 27.852762976298393]}

## Binary search (step 2) starts
Candidate diff: 0.0625000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8485849, upper bound: 27.8470298
time: 0.69 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8465972, upper bound: 27.8465972
time: 0.66 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.56 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.56
Output dim: 0, lower bound: -27.8485849, upper bound: 27.8470298
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.56
Output dim: 0, lower bound: -27.8465972, upper bound: 27.8465972

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -4.8150167, 17.2120781, -5.7017121, 19.8065739, -24.6215839, 22.9137897
1: -7.9723043, 17.4862099, -9.3930426, 20.2009811, -28.1732864, 26.8792534
2: -6.4574275, 18.9246101, -7.6706657, 21.8329620, -28.2903900, 26.5952759
3: -7.0645366, 26.1020927, -8.3439598, 30.0234928, -37.0880280, 34.4460526
4: -5.6939859, 24.4059410, -6.7792501, 28.3508034, -34.0447884, 31.1851845

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8060786, upper bound: 27.8038854
time: 0.86 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7817701, upper bound: 27.7829189
time: 1.38 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -6.2154808, 21.2638779, -6.6680698, 22.5731068, -28.7885876, 27.9319458
1: -10.2115059, 21.7044182, -10.9209986, 23.0753460, -33.2868500, 32.6254158
2: -8.3589554, 23.4411602, -8.9612103, 24.8656731, -33.2246246, 32.4023666
3: -9.0649576, 32.1635170, -9.6909332, 34.1161194, -43.1810722, 41.8544464
4: -7.4015422, 30.4624748, -7.9296932, 32.3294716, -39.7310143, 38.3921661

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7900049, upper bound: 27.7850681
time: 0.74 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7601391, upper bound: 27.7601391
time: 0.81 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.21 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.21
Output dim: 0, lower bound: -27.8060786, upper bound: 27.8038854
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.21
Output dim: 0, lower bound: -27.7817701, upper bound: 27.7829189
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.21
Output dim: 0, lower bound: -27.7900049, upper bound: 27.7850681
IS_A2_B2, status: Status.VERIFIED, split count: 2, time: 4.21
Output dim: 0, lower bound: -27.7601391, upper bound: 27.7601391

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -4.5416622, 16.3911209, -4.9800100, 17.5521507, -22.0938072, 21.3711281
1: -7.5367661, 16.6260834, -8.2413111, 17.8693676, -25.4061337, 24.8673935
2: -6.0881882, 18.0136566, -6.6934099, 19.3738976, -25.4620838, 24.7070618
3: -6.6800528, 24.8602180, -7.3300457, 26.6334858, -33.3135338, 32.1902618
4: -5.3691096, 23.1925087, -5.9270549, 25.1043282, -30.4734325, 29.1195641

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7817701, upper bound: 27.7829189
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7817701, upper bound: 27.7829189
time: 1.00 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -4.3502960, 15.7953491, -5.8211989, 20.5492229, -24.8995190, 21.6165485
1: -7.2296133, 16.0116482, -9.5734949, 20.8136292, -28.0432434, 25.5851440
2: -5.8287868, 17.3624058, -7.8012676, 22.5619774, -28.3907642, 25.1636696
3: -6.4112043, 23.9572716, -8.4682245, 31.0174236, -37.4286270, 32.4254951
4: -5.1430759, 22.3230343, -6.9039507, 29.1375294, -34.2805977, 29.2269859

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7817701, upper bound: 27.7829189
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7817701, upper bound: 27.7829189
time: 0.65 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -5.9067326, 20.3583755, -5.8470201, 20.1596336, -26.0663662, 26.2053947
1: -9.7248631, 20.7617283, -9.6252575, 20.5580883, -30.2829514, 30.3869858
2: -7.9437990, 22.4499512, -7.8564663, 22.2309933, -30.1747932, 30.3064175
3: -8.6380367, 30.8078308, -8.5549374, 30.4993610, -39.1373978, 39.3627701
4: -7.0395150, 29.1613369, -6.9642706, 28.8633003, -35.9028168, 36.1256065

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7435087, upper bound: 27.7324940
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7601391, upper bound: 27.7601391
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7601391, upper bound: 27.7601391
time: 0.93 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 5.48 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.48
Output dim: 0, lower bound: -27.7817701, upper bound: 27.7829189
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.48
Output dim: 0, lower bound: -27.7817701, upper bound: 27.7829189
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.48
Output dim: 0, lower bound: -27.7817701, upper bound: 27.7829189
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.48
Output dim: 0, lower bound: -27.7817701, upper bound: 27.7829189
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 5.48
Output dim: 0, lower bound: -27.7601391, upper bound: 27.7601391
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 5.48
Output dim: 0, lower bound: -27.7601391, upper bound: 27.7601391

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4.1239586, 15.1485586, -4.9800100, 17.5521507, -21.6761093, 20.1285667
1: -6.8710823, 15.3317356, -8.2413111, 17.8693676, -24.7404499, 23.5730476
2: -5.5255399, 16.6497955, -6.6934099, 19.3738976, -24.8994370, 23.3432026
3: -6.0924091, 22.9905643, -7.3300457, 26.6334858, -32.7258949, 30.3206062
4: -4.8740869, 21.3502178, -5.9270549, 25.1043282, -29.9784145, 27.2772732

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8060786, upper bound: 27.8038854
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8060786, upper bound: 27.8038854
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -5.2290773, 18.9489193, -4.9800100, 17.5521507, -22.7812252, 23.9289284
1: -8.6279545, 19.1228676, -8.2413111, 17.8693676, -26.4973221, 27.3641777
2: -6.9899049, 20.7296162, -6.6934099, 19.3738976, -26.3637981, 27.4230251
3: -7.5971055, 28.6222916, -7.3300457, 26.6334858, -34.2305908, 35.9523354
4: -6.1555519, 26.5859127, -5.9270549, 25.1043282, -31.2598763, 32.5129662

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8060786, upper bound: 27.8038854
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8060786, upper bound: 27.8038854
time: 1.10 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4.1239586, 15.1485586, -5.8211989, 20.5492229, -24.6731796, 20.9697533
1: -6.8710823, 15.3317356, -9.5734949, 20.8136292, -27.6847115, 24.9052315
2: -5.5255399, 16.6497955, -7.8012676, 22.5619774, -28.0875168, 24.4510612
3: -6.0924091, 22.9905643, -8.4682245, 31.0174236, -37.1098328, 31.4587860
4: -4.8740869, 21.3502178, -6.9039507, 29.1375294, -34.0116005, 28.2541676

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7817701, upper bound: 27.7829189
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7817701, upper bound: 27.7829189
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -5.2223582, 18.9270706, -5.8211989, 20.5492229, -25.7715816, 24.7482700
1: -8.6170578, 19.1002598, -9.5734949, 20.8136292, -29.4306870, 28.6737556
2: -6.9805036, 20.7045078, -7.8012676, 22.5619774, -29.5424786, 28.5057755
3: -7.5865173, 28.5869350, -8.4682245, 31.0174236, -38.6039429, 37.0551567
4: -6.1469746, 26.5502129, -6.9039507, 29.1375294, -35.2844963, 33.4541626

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7817701, upper bound: 27.7829189
time: 1.20 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7817701, upper bound: 27.7829189
time: 1.14 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 5.25 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.25
Output dim: 0, lower bound: -27.8060786, upper bound: 27.8038854
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.25
Output dim: 0, lower bound: -27.8060786, upper bound: 27.8038854
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.25
Output dim: 0, lower bound: -27.8060786, upper bound: 27.8038854
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.25
Output dim: 0, lower bound: -27.8060786, upper bound: 27.8038854
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.25
Output dim: 0, lower bound: -27.7817701, upper bound: 27.7829189
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.25
Output dim: 0, lower bound: -27.7817701, upper bound: 27.7829189
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.25
Output dim: 0, lower bound: -27.7817701, upper bound: 27.7829189
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.25
Output dim: 0, lower bound: -27.7817701, upper bound: 27.7829189

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -4.1239586, 15.1485586, -4.1239586, 15.1485586, -19.2725143, 19.2725124
1: -6.8710823, 15.3317356, -6.8710823, 15.3317356, -22.2028179, 22.2028179
2: -5.5255399, 16.6497955, -5.5255399, 16.6497955, -22.1753349, 22.1753349
3: -6.0924091, 22.9905643, -6.0924091, 22.9905643, -29.0829735, 29.0829735
4: -4.8740869, 21.3502178, -4.8740869, 21.3502178, -26.2243042, 26.2243042

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8232525, upper bound: 27.8181133
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8485849, upper bound: 27.8470298
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -4.1239586, 15.1485586, -5.3985829, 18.8200378, -22.9439964, 20.5471382
1: -6.8710823, 15.3317356, -8.9148941, 19.1645012, -26.0355835, 24.2466297
2: -5.5255399, 16.6497955, -7.2588787, 20.7659760, -26.2915134, 23.9086723
3: -6.0924091, 22.9905643, -7.9256678, 28.5038872, -34.5962868, 30.9162292
4: -4.8740869, 21.3502178, -6.4428453, 26.9448719, -31.8189545, 27.7930641

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8232525, upper bound: 27.8181133
time: 1.15 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8485849, upper bound: 27.8470298
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -5.2290773, 18.9489193, -4.1239586, 15.1485586, -20.3776302, 23.0728779
1: -8.6279545, 19.1228676, -6.8710823, 15.3317356, -23.9596901, 25.9939499
2: -6.9899049, 20.7296162, -5.5255399, 16.6497955, -23.6396961, 26.2551556
3: -7.5971055, 28.6222916, -6.0924091, 22.9905643, -30.5876656, 34.7146988
4: -6.1555519, 26.5859127, -4.8740869, 21.3502178, -27.5057697, 31.4599972

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.1773425, upper bound: 27.2234041
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8060786, upper bound: 27.8038854
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -5.2290773, 18.9489193, -5.3985829, 18.8200378, -24.0491142, 24.3475018
1: -8.6279545, 19.1228676, -8.9148941, 19.1645012, -27.7924557, 28.0377617
2: -6.9899049, 20.7296162, -7.2588787, 20.7659760, -27.7558765, 27.9884949
3: -7.5971055, 28.6222916, -7.9256678, 28.5038872, -36.1009789, 36.5479546
4: -6.1555519, 26.5859127, -6.4428453, 26.9448719, -33.1004181, 33.0287552

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.1773425, upper bound: 27.2234041
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8060786, upper bound: 27.8038854
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -4.1239586, 15.1485586, -5.3676920, 19.4284458, -23.5524025, 20.5162487
1: -6.8710823, 15.3317356, -8.8535709, 19.6024017, -26.4734840, 24.1853065
2: -5.5255399, 16.6497955, -7.1736655, 21.2490845, -26.7746239, 23.8234596
3: -6.0924091, 22.9905643, -7.7958689, 29.3602962, -35.4527054, 30.7864265
4: -4.8740869, 21.3502178, -6.3186235, 27.2952156, -32.1692963, 27.6688385

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -26.9979183, upper bound: 26.9632221
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7875358, upper bound: 27.7901622
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4.1239586, 15.1485586, -6.2443461, 21.8634129, -25.9873714, 21.3929005
1: -6.8710823, 15.3317356, -10.2460995, 22.1588764, -29.0299587, 25.5778351
2: -5.5255399, 16.6497955, -8.3748341, 23.9880085, -29.5135460, 25.0246239
3: -6.0924091, 22.9905643, -9.0668802, 32.9503479, -39.0427551, 32.0574455
4: -4.8740869, 21.3502178, -7.4148474, 31.0294037, -35.9034843, 28.7650642

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -26.9979183, upper bound: 26.9632221
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7875358, upper bound: 27.7901622
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -5.2223582, 18.9270706, -5.3676920, 19.4284458, -24.6508026, 24.2947617
1: -8.6170578, 19.1002598, -8.8535709, 19.6024017, -28.2194595, 27.9538307
2: -6.9805036, 20.7045078, -7.1736655, 21.2490845, -28.2295837, 27.8781738
3: -7.5865173, 28.5869350, -7.7958689, 29.3602962, -36.9468155, 36.3827972
4: -6.1469746, 26.5502129, -6.3186235, 27.2952156, -33.4421844, 32.8688354

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.0035380, upper bound: 26.9666982
time: 1.13 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7817701, upper bound: 27.7829189
time: 1.21 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -5.2223582, 18.9270706, -6.2443461, 21.8634129, -27.0857697, 25.1714172
1: -8.6170578, 19.1002598, -10.2460995, 22.1588764, -30.7759342, 29.3463573
2: -6.9805036, 20.7045078, -8.3748341, 23.9880085, -30.9685116, 29.0793381
3: -7.5865173, 28.5869350, -9.0668802, 32.9503479, -40.5368652, 37.6538162
4: -6.1469746, 26.5502129, -7.4148474, 31.0294037, -37.1763763, 33.9650612

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.0035380, upper bound: 26.9667088
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7817701, upper bound: 27.7829189
time: 0.89 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 5.92 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.92
Output dim: 0, lower bound: -27.8232525, upper bound: 27.8181133
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.92
Output dim: 0, lower bound: -27.8485849, upper bound: 27.8470298
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.92
Output dim: 0, lower bound: -27.8232525, upper bound: 27.8181133
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.92
Output dim: 0, lower bound: -27.8485849, upper bound: 27.8470298
IS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.92
Output dim: 0, lower bound: -27.1773425, upper bound: 27.2234041
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.92
Output dim: 0, lower bound: -27.8060786, upper bound: 27.8038854
IS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 5.92
Output dim: 0, lower bound: -27.1773425, upper bound: 27.2234041
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.92
Output dim: 0, lower bound: -27.8060786, upper bound: 27.8038854
IS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.92
Output dim: 0, lower bound: -26.9979183, upper bound: 26.9632221
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.92
Output dim: 0, lower bound: -27.7875358, upper bound: 27.7901622
IS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 5.92
Output dim: 0, lower bound: -26.9979183, upper bound: 26.9632221
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.92
Output dim: 0, lower bound: -27.7875358, upper bound: 27.7901622
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.92
Output dim: 0, lower bound: -27.0035380, upper bound: 26.9666982
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.92
Output dim: 0, lower bound: -27.7817701, upper bound: 27.7829189
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 5.92
Output dim: 0, lower bound: -27.0035380, upper bound: 26.9667088
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.92
Output dim: 0, lower bound: -27.7817701, upper bound: 27.7829189

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -3.9672580, 14.7736979, -3.9205022, 14.5063133, -18.4735699, 18.6941986
1: -6.6120090, 14.9280148, -6.5444918, 14.6787481, -21.2907543, 21.4725075
2: -5.3131099, 16.2373619, -5.2503572, 15.9511948, -21.2643051, 21.4877167
3: -5.8528652, 22.4279957, -5.8058863, 22.0311432, -27.8840065, 28.2338810
4: -4.6880665, 20.7485371, -4.6334066, 20.4118900, -25.0999565, 25.3819427

Time for backsubstitution: 2.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8509207, upper bound: 27.8504971
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8509207, upper bound: 27.8502625
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -4.0236449, 14.8443575, -4.1088710, 15.1023540, -19.1259918, 18.9532261
1: -6.7107162, 15.0167494, -6.8468270, 15.2836828, -21.9943962, 21.8635750
2: -5.3908873, 16.3171444, -5.5053096, 16.5992889, -21.9901772, 21.8224545
3: -5.9503188, 22.5346546, -6.0708151, 22.9211502, -28.8714695, 28.6054688
4: -4.7566547, 20.9033737, -4.8561678, 21.2817974, -26.0384521, 25.7595406

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8520051, upper bound: 27.8520137
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8520051, upper bound: 27.8520137
time: 1.05 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -3.9672580, 14.7736979, -5.1723585, 18.0948391, -22.0620975, 19.9460564
1: -6.6120090, 14.9280148, -8.5507574, 18.4211140, -25.0331230, 23.4787712
2: -5.3131099, 16.2373619, -6.9533949, 19.9736919, -25.2868023, 23.1907520
3: -5.8528652, 22.4279957, -7.6029210, 27.4189377, -33.2718010, 30.0309162
4: -4.6880665, 20.7485371, -6.1693735, 25.8902683, -30.5783272, 26.9179115

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8020399, upper bound: 27.8058861
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8114254, upper bound: 27.8090971
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -4.0236449, 14.8443575, -5.3816128, 18.7673569, -22.7910004, 20.2259636
1: -6.7107162, 15.0167494, -8.8877287, 19.1103306, -25.8210449, 23.9044781
2: -5.3908873, 16.3171444, -7.2361555, 20.7084465, -26.0993347, 23.5532990
3: -5.9503188, 22.5346546, -7.9016228, 28.4247894, -34.3751068, 30.4362774
4: -4.7566547, 20.9033737, -6.4230556, 26.8685741, -31.6252270, 27.3264294

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7662412, upper bound: 27.7623216
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7662412, upper bound: 27.8470298
time: 1.08 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -5.1222334, 18.6311798, -4.1239586, 15.1485586, -20.2707901, 22.7551384
1: -8.4574766, 18.7960930, -6.8710823, 15.3317356, -23.7892113, 25.6671734
2: -6.8458257, 20.3818703, -5.5255399, 16.6497955, -23.4956207, 25.9074097
3: -7.4456444, 28.1472301, -6.0924091, 22.9905643, -30.4362030, 34.2396393
4: -6.0291471, 26.1222534, -4.8740869, 21.3502178, -27.3793640, 30.9963379

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7944433, upper bound: 27.7897170
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8121562, upper bound: 27.8108170
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -5.1222334, 18.6311798, -5.3985829, 18.8200378, -23.9422722, 24.0297623
1: -8.4574766, 18.7960930, -8.9148941, 19.1645012, -27.6219788, 27.7109833
2: -6.8458257, 20.3818703, -7.2588787, 20.7659760, -27.6118011, 27.6407490
3: -7.4456444, 28.1472301, -7.9256678, 28.5038872, -35.9495125, 36.0728951
4: -6.0291471, 26.1222534, -6.4428453, 26.9448719, -32.9740143, 32.5650940

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.2834678, upper bound: 27.3373241
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.2846718, upper bound: 27.3383870
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -4.0084338, 14.8078671, -5.3676920, 19.4284458, -23.4368782, 20.1755600
1: -6.6869602, 14.9791889, -8.8535709, 19.6024017, -26.2893620, 23.8327599
2: -5.3691125, 16.2761097, -7.1736655, 21.2490845, -26.6181946, 23.4497757
3: -5.9292288, 22.4815655, -7.7958689, 29.3602962, -35.2895241, 30.2774353
4: -4.7361321, 20.8426514, -6.3186235, 27.2952156, -32.0313454, 27.1612740

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7201440, upper bound: 27.7305941
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7201440, upper bound: 27.8121562
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -4.0084338, 14.8078671, -6.2443461, 21.8634129, -25.8718472, 21.0522137
1: -6.6869602, 14.9791889, -10.2460995, 22.1588764, -28.8458366, 25.2252865
2: -5.3691125, 16.2761097, -8.3748341, 23.9880085, -29.3571205, 24.6509418
3: -5.9292288, 22.4815655, -9.0668802, 32.9503479, -38.8795776, 31.5484467
4: -4.7361321, 20.8426514, -7.4148474, 31.0294037, -35.7655373, 28.2574997

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.2406616, upper bound: 27.3009264
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.1033350, upper bound: 27.0486369
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -5.1178064, 18.6168690, -5.3676920, 19.4284458, -24.5462532, 23.9845619
1: -8.4503164, 18.7812614, -8.8535709, 19.6024017, -28.0527172, 27.6348305
2: -6.8396845, 20.3654137, -7.1736655, 21.2490845, -28.0887661, 27.5390797
3: -7.4386759, 28.1241074, -7.7958689, 29.3602962, -36.7989731, 35.9199753
4: -6.0235567, 26.0988503, -6.3186235, 27.2952156, -33.3187675, 32.4174728

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.6754993, upper bound: 27.6767816
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.6754993, upper bound: 27.8045490
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -5.1178064, 18.6168690, -6.2443461, 21.8634129, -26.9812202, 24.8612156
1: -8.4503164, 18.7812614, -10.2460995, 22.1588764, -30.6091919, 29.0273590
2: -6.8396845, 20.3654137, -8.3748341, 23.9880085, -30.8276863, 28.7402420
3: -7.4386759, 28.1241074, -9.0668802, 32.9503479, -40.3890228, 37.1909866
4: -6.0235567, 26.0988503, -7.4148474, 31.0294037, -37.0529594, 33.5136948

Time for backsubstitution: 2.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.2511649, upper bound: 27.3078457
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.1078456, upper bound: 27.0521568
time: 0.74 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 6.21 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -27.8509207, upper bound: 27.8504971
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -27.8509207, upper bound: 27.8502625
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -27.8520051, upper bound: 27.8520137
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -27.8520051, upper bound: 27.8520137
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -27.8020399, upper bound: 27.8058861
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -27.8114254, upper bound: 27.8090971
IS_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 6.21
Output dim: 0, lower bound: -27.7662412, upper bound: 27.7623216
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -27.7662412, upper bound: 27.8470298
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -27.7944433, upper bound: 27.7897170
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -27.8121562, upper bound: 27.8108170
IS_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 6.21
Output dim: 0, lower bound: -27.2834678, upper bound: 27.3373241
IS_A1_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 6.21
Output dim: 0, lower bound: -27.2846718, upper bound: 27.3383870
IS_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 6.21
Output dim: 0, lower bound: -27.7201440, upper bound: 27.7305941
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -27.7201440, upper bound: 27.8121562
IS_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 6.21
Output dim: 0, lower bound: -27.2406616, upper bound: 27.3009264
IS_A1_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 6.21
Output dim: 0, lower bound: -27.1033350, upper bound: 27.0486369
IS_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 6.21
Output dim: 0, lower bound: -27.6754993, upper bound: 27.6767816
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.21
Output dim: 0, lower bound: -27.6754993, upper bound: 27.8045490
IS_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 6.21
Output dim: 0, lower bound: -27.2511649, upper bound: 27.3078457
IS_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 6.21
Output dim: 0, lower bound: -27.1078456, upper bound: 27.0521568

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -3.5195677, 13.3965511, -3.4101064, 12.9385662, -16.4581337, 16.8066540
1: -5.9020076, 13.5197325, -5.7202034, 13.0840054, -18.9860134, 19.2399349
2: -4.7059097, 14.7364149, -4.5616255, 14.2754860, -18.9813957, 19.2980404
3: -5.2278919, 20.3638210, -5.0716677, 19.6485672, -24.8764591, 25.4354877
4: -4.1621838, 18.7631073, -4.0424166, 18.1593094, -22.3214931, 22.8055229

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8509207, upper bound: 27.8502625
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8509207, upper bound: 27.8502625
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -3.9672580, 14.7736979, -3.8078008, 14.1757669, -18.1430244, 18.5814991
1: -6.6120090, 14.9280148, -6.3650160, 14.3370047, -20.9490128, 21.2930298
2: -5.3131099, 16.2373619, -5.0976214, 15.5889521, -20.9020596, 21.3349781
3: -5.8528652, 22.4279957, -5.6470871, 21.5381603, -27.3910236, 28.0750809
4: -4.6880665, 20.7485371, -4.4992003, 19.9197578, -24.6078243, 25.2477379

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8509207, upper bound: 27.8502625
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8509207, upper bound: 27.8502625
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -4.0236449, 14.8443575, -3.9672580, 14.7736979, -18.7973366, 18.8116150
1: -6.7107162, 15.0167494, -6.6120090, 14.9280148, -21.6387310, 21.6287575
2: -5.3908873, 16.3171444, -5.3131099, 16.2373619, -21.6282501, 21.6302547
3: -5.9503188, 22.5346546, -5.8528652, 22.4279957, -28.3783150, 28.3875179
4: -4.7566547, 20.9033737, -4.6880665, 20.7485371, -25.5051918, 25.5914402

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8504971, upper bound: 27.8509207
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8502625, upper bound: 27.8509207
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -4.0236449, 14.8443575, -4.0236449, 14.8443575, -18.8679943, 18.8679943
1: -6.7107162, 15.0167494, -6.7107162, 15.0167494, -21.7274647, 21.7274628
2: -5.3908873, 16.3171444, -5.3908873, 16.3171444, -21.7080307, 21.7080307
3: -5.9503188, 22.5346546, -5.9503188, 22.5346546, -28.4849739, 28.4849739
4: -4.7566547, 20.9033737, -4.7566547, 20.9033737, -25.6600285, 25.6600285

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8504971, upper bound: 27.8517792
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8502625, upper bound: 27.8517792
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -3.6227920, 13.7032461, -4.0750151, 14.6094208, -18.2322121, 17.7782593
1: -6.0595536, 13.8466091, -6.7990160, 14.8599825, -20.9195309, 20.6456261
2: -4.8494225, 15.0744028, -5.4651332, 16.1577816, -21.0072041, 20.5395355
3: -5.3798299, 20.8323956, -6.0782089, 22.2382603, -27.6180878, 26.9106045
4: -4.2806849, 19.2089672, -4.8471551, 20.8717060, -25.1523914, 24.0561218

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8020399, upper bound: 27.8028684
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8020399, upper bound: 27.8058861
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -3.8935475, 14.5632219, -4.9232349, 17.3180504, -21.2115936, 19.4864540
1: -6.4948845, 14.7112169, -8.1393394, 17.6339283, -24.1288128, 22.8505535
2: -5.2134008, 16.0065613, -6.6223621, 19.1261806, -24.3395805, 22.6289234
3: -5.7509041, 22.1126175, -7.2486773, 26.2315121, -31.9824162, 29.3612938
4: -4.5996795, 20.4373188, -5.8781166, 24.7368813, -29.3365574, 26.3154354

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7141053, upper bound: 27.7112872
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8097473, upper bound: 27.8065367
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8114254, upper bound: 27.8090971
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4.0236449, 14.8443575, -5.2741375, 18.4337215, -22.4573593, 20.1184959
1: -6.7107162, 15.0167494, -8.7155638, 18.7671432, -25.4778576, 23.7323132
2: -5.3908873, 16.3171444, -7.0923862, 20.3441086, -25.7349949, 23.4095306
3: -5.9503188, 22.5346546, -7.7492294, 27.9237080, -33.8740234, 30.2838802
4: -4.7566547, 20.9033737, -6.2977266, 26.3849220, -31.1415768, 27.2010975

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.6485557, upper bound: 27.6435856
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7662412, upper bound: 27.8465972
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -4.8936067, 17.8978958, -3.9672580, 14.7736979, -19.6673050, 21.8651543
1: -8.0921526, 18.0519333, -6.6120090, 14.9280148, -23.0201683, 24.6639423
2: -6.5377674, 19.5876312, -5.3131099, 16.2373619, -22.7751274, 24.9007416
3: -7.1217885, 27.0530205, -5.8528652, 22.4279957, -29.5497837, 32.9058838
4: -5.7605848, 25.0699158, -4.6880665, 20.7485371, -26.5091209, 29.7579823

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7651004, upper bound: 27.7575623
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7944433, upper bound: 27.7897170
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -5.1083002, 18.5880833, -4.0236449, 14.8443575, -19.9526577, 22.6117229
1: -8.4351006, 18.7514839, -6.7107162, 15.0167494, -23.4518490, 25.4622002
2: -6.8271780, 20.3347321, -5.3908873, 16.3171444, -23.1443214, 25.7256203
3: -7.4256735, 28.0824432, -5.9503188, 22.5346546, -29.9603233, 34.0327606
4: -6.0127130, 26.0595322, -4.7566547, 20.9033737, -26.9160843, 30.8161831

Time for backsubstitution: 2.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7255403, upper bound: 27.7600246
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8121562, upper bound: 27.8108170
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -4.0084338, 14.8078671, -5.2453480, 19.0520000, -23.0604343, 20.0532131
1: -6.6869602, 14.9791889, -8.6571541, 19.2172604, -25.9042206, 23.6363430
2: -5.3691125, 16.2761097, -7.0090017, 20.8379440, -26.2070560, 23.2851105
3: -5.9292288, 22.4815655, -7.6216717, 28.7943916, -34.7236214, 30.1032372
4: -4.7361321, 20.8426514, -6.1741533, 26.7458191, -31.4819508, 27.0167999

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7199693, upper bound: 27.7696827
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.7201440, upper bound: 27.8104692
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -5.1178064, 18.6168690, -5.2453480, 19.0520000, -24.1698074, 23.8622150
1: -8.4503164, 18.7812614, -8.6571541, 19.2172604, -27.6675758, 27.4384155
2: -6.8396845, 20.3654137, -7.0090017, 20.8379440, -27.6776276, 27.3744164
3: -7.4386759, 28.1241074, -7.6216717, 28.7943916, -36.2330666, 35.7457695
4: -6.0235567, 26.0988503, -6.1741533, 26.7458191, -32.7693748, 32.2730026

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -26.5687516, upper bound: 26.5375071
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.6754993, upper bound: 27.8045489
time: 1.19 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 6.72 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.72
Output dim: 0, lower bound: -27.8509207, upper bound: 27.8502625
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.72
Output dim: 0, lower bound: -27.8509207, upper bound: 27.8502625
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.72
Output dim: 0, lower bound: -27.8509207, upper bound: 27.8502625
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.72
Output dim: 0, lower bound: -27.8509207, upper bound: 27.8502625
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.72
Output dim: 0, lower bound: -27.8504971, upper bound: 27.8509207
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.72
Output dim: 0, lower bound: -27.8502625, upper bound: 27.8509207
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.72
Output dim: 0, lower bound: -27.8504971, upper bound: 27.8517792
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.72
Output dim: 0, lower bound: -27.8502625, upper bound: 27.8517792
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.72
Output dim: 0, lower bound: -27.8020399, upper bound: 27.8028684
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.72
Output dim: 0, lower bound: -27.8020399, upper bound: 27.8058861
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.72
Output dim: 0, lower bound: -27.8097473, upper bound: 27.8065367
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.72
Output dim: 0, lower bound: -27.8114254, upper bound: 27.8090971
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.72
Output dim: 0, lower bound: -27.6485557, upper bound: 27.6435856
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.72
Output dim: 0, lower bound: -27.7662412, upper bound: 27.8465972
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.72
Output dim: 0, lower bound: -27.7651004, upper bound: 27.7575623
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.72
Output dim: 0, lower bound: -27.7944433, upper bound: 27.7897170
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.72
Output dim: 0, lower bound: -27.7255403, upper bound: 27.7600246
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.72
Output dim: 0, lower bound: -27.8121562, upper bound: 27.8108170
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.72
Output dim: 0, lower bound: -27.7199693, upper bound: 27.7696827
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.72
Output dim: 0, lower bound: -27.7201440, upper bound: 27.8104692
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.72
Output dim: 0, lower bound: -26.5687516, upper bound: 26.5375071
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.72
Output dim: 0, lower bound: -27.6754993, upper bound: 27.8045489

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -3.3273158, 12.8372402, -3.4101064, 12.9385662, -16.2658825, 16.2473431
1: -5.6021156, 12.9728231, -5.7202034, 13.0840054, -18.6861210, 18.6930237
2: -4.4462528, 14.1458454, -4.5616255, 14.2754860, -18.7217388, 18.7074699
3: -4.9587526, 19.5276680, -5.0716677, 19.6485672, -24.6073189, 24.5993328
4: -3.9472556, 18.0581570, -4.0424166, 18.1593094, -22.1065655, 22.1005745

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8393047, upper bound: 27.8380695
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7458159, upper bound: 27.7553412
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -3.8444366, 14.4183207, -3.4101064, 12.9385662, -16.7830009, 17.8284245
1: -6.4171948, 14.5603256, -5.7202034, 13.0840054, -19.5011997, 20.2805271
2: -5.1471019, 15.8479376, -4.5616255, 14.2754860, -19.4225883, 20.4095631
3: -5.6811781, 21.9004230, -5.0716677, 19.6485672, -25.3297424, 26.9720879
4: -4.5425673, 20.2274742, -4.0424166, 18.1593094, -22.7018776, 24.2698898

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8393047, upper bound: 27.8425077
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -27.7458159, upper bound: 27.7553412
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -3.3273158, 12.8372402, -3.8078008, 14.1757669, -17.5030823, 16.6450405
1: -5.6021156, 12.9728231, -6.3650160, 14.3370047, -19.9391212, 19.3378391
2: -4.4462528, 14.1458454, -5.0976214, 15.5889521, -20.0352020, 19.2434635
3: -4.9587526, 19.5276680, -5.6470871, 21.5381603, -26.4969139, 25.1747532
4: -3.9472556, 18.0581570, -4.4992003, 19.9197578, -23.8670139, 22.5573559

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8494041, upper bound: 27.8494041
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8494041, upper bound: 27.8502625
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -3.8642793, 14.4800587, -3.8078008, 14.1757669, -18.0400467, 18.2878590
1: -6.4491496, 14.6237555, -6.3650160, 14.3370047, -20.7861538, 20.9887714
2: -5.1740036, 15.9146786, -5.0976214, 15.5889521, -20.7629509, 21.0122967
3: -5.7092237, 21.9910412, -5.6470871, 21.5381603, -27.2473831, 27.6381245
4: -4.5661373, 20.3150978, -4.4992003, 19.9197578, -24.4858952, 24.8142986

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8494041, upper bound: 27.8494041
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8494041, upper bound: 27.8502625
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -3.5486126, 13.3772373, -3.5195677, 13.3965511, -16.9451580, 16.8968048
1: -5.9423294, 13.5300770, -5.9020076, 13.5197325, -19.4620628, 19.4320831
2: -4.7485104, 14.7520132, -4.7059097, 14.7364149, -19.4849243, 19.4579239
3: -5.2675343, 20.3012791, -5.2278919, 20.3638210, -25.6313553, 25.5291710
4: -4.2054482, 18.7917194, -4.1621838, 18.7631073, -22.9685535, 22.9539032

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8502625, upper bound: 27.8509207
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8502625, upper bound: 27.8509207
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -3.9118266, 14.5160675, -3.9672580, 14.7736979, -18.6855221, 18.4833260
1: -6.5327997, 14.6775446, -6.6120090, 14.9280148, -21.4608154, 21.2895527
2: -5.2392712, 15.9570942, -5.3131099, 16.2373619, -21.4766312, 21.2702045
3: -5.7927489, 22.0449142, -5.8528652, 22.4279957, -28.2207432, 27.8977776
4: -4.6233401, 20.4146690, -4.6880665, 20.7485371, -25.3718777, 25.1027355

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8502625, upper bound: 27.8509207
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8502625, upper bound: 27.8509207
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -3.5486126, 13.3772373, -3.5663493, 13.4350529, -16.9836636, 16.9435863
1: -5.9423294, 13.5300770, -5.9853830, 13.5739326, -19.5162621, 19.5154572
2: -4.7485104, 14.7520132, -4.7705522, 14.7806654, -19.5291748, 19.5225658
3: -5.2675343, 20.3012791, -5.3106637, 20.4229565, -25.6904907, 25.6119423
4: -4.2054482, 18.7917194, -4.2185721, 18.8775692, -23.0830173, 23.0102921

Time for backsubstitution: 2.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8517792, upper bound: 27.8517792
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8517792, upper bound: 27.8517792
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -3.9118266, 14.5160675, -4.0236449, 14.8443575, -18.7561798, 18.5397072
1: -6.5327997, 14.6775446, -6.7107162, 15.0167494, -21.5495472, 21.3882580
2: -5.2392712, 15.9570942, -5.3908873, 16.3171444, -21.5564137, 21.3479805
3: -5.7927489, 22.0449142, -5.9503188, 22.5346546, -28.3274021, 27.9952335
4: -4.6233401, 20.4146690, -4.7566547, 20.9033737, -25.5267124, 25.1713238

Time for backsubstitution: 2.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8517792, upper bound: 27.8517792
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -27.8517792, upper bound: 27.8517792
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -3.4710240, 13.2884607, -4.0750151, 14.6094208, -18.0804443, 17.3634739
1: -5.8060937, 13.4011593, -6.7990160, 14.8599825, -20.6660748, 20.2001762
2: -4.6429787, 14.5897045, -5.4651332, 16.1577816, -20.8007545, 20.0548344
3: -5.1681485, 20.1465702, -6.0782089, 22.2382603, -27.4064083, 26.2247791
4: -4.0871625, 18.5050297, -4.8471551, 20.8717060, -24.9588680, 23.3521843

Time for backsubstitution: 2.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=30.45956039428711
rel_dist={0: [-27.852403738376353, 27.852403738376353]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 1110.07 seconds
