## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_4.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 886.64361740241


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742)
1: (-437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521)
2: (-439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498)
3: (-536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455)
4: (-473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094)

## BASE Result
execution time: IAR + LP analysis = 1.68 + 2.04 = 3.72 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -886.6916286, upper bound: 886.6916286


# Binary Search by BASE starts (time budget: 1196.28 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=1024.72607421875
rel_dist={0: [-886.6916286331143, 886.6916286331141]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=1024.72607421875
rel_dist={0: [-886.6909129349328, 886.6909129349328]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=1024.72607421875
rel_dist={0: [-886.6885385920837, 886.6885385920837]}

## Binary search (step 3) starts
Candidate diff: 0.0625000


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=1024.72607421875
rel_dist={0: [-886.6867135844589, 886.6867135844589]}

## Binary search (step 4) starts
Candidate diff: 0.0312500


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=1024.72607421875
rel_dist={0: [-886.6855568160574, 886.6855568160574]}

## Binary search (step 5) starts
Candidate diff: 0.0156250


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=1024.72607421875
rel_dist={0: [-886.6846760554688, 886.6846760554686]}

## Binary search (step 6) starts
Candidate diff: 0.0078125


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0078125, mid=0.0078125, abs_max=1024.72607421875
rel_dist={0: [-886.6841389728621, 886.684138972862]}

## Binary search (step 7) starts
Candidate diff: 0.0039062


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0039062, mid=0.0039062, abs_max=1024.72607421875
rel_dist={0: [-886.6838666122214, 886.6838666122214]}

## Binary search (step 8) starts
Candidate diff: 0.0019531


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0019531, mid=0.0019531, abs_max=1024.72607421875
rel_dist={0: [-886.6837304319017, 886.6837304319015]}

## Binary search (step 9) starts
Candidate diff: 0.0009766


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0009766, mid=0.0009766, abs_max=1024.72607421875
rel_dist={0: [-886.6836623417433, 886.6836623417435]}

## Binary search (step 10) starts
Candidate diff: 0.0004883


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0004883, mid=0.0004883, abs_max=1024.72607421875
rel_dist={0: [-886.6836282966669, 886.6836282966672]}

## Binary search (step 11) starts
Candidate diff: 0.0002441


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0002441, mid=0.0002441, abs_max=1024.72607421875
rel_dist={0: [-886.6836112741353, 886.683611274135]}

## Binary search (step 12) starts
Candidate diff: 0.0001221


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0001221, mid=0.0001221, abs_max=1024.72607421875
rel_dist={0: [-886.6836027628814, 886.683602762881]}

## Binary search (step 13) starts
Candidate diff: 0.0000610


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000610, mid=0.0000610, abs_max=1024.72607421875
rel_dist={0: [-886.6835985072787, 886.6835985072787]}

## Binary search (step 14) starts
Candidate diff: 0.0000305


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000305, mid=0.0000305, abs_max=1024.72607421875
rel_dist={0: [-886.6835963802994, 886.6835963795252]}

## Binary search (step 15) starts
Candidate diff: 0.0000153


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000153, mid=0.0000153, abs_max=1024.72607421875
rel_dist={0: [-886.6835953040254, 886.6835953042985]}

## Binary search (step 16) starts
Candidate diff: 0.0000076


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000076, mid=0.0000076, abs_max=1024.72607421875
rel_dist={0: [-886.6835947653071, 886.6835947653071]}

## Binary search (step 17) starts
Candidate diff: 0.0000038


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000038, mid=0.0000038, abs_max=1024.72607421875
rel_dist={0: [-886.6835945041508, 886.6835944963228]}

## Binary search (step 18) starts
Candidate diff: 0.0000019


## IAR start
Binary search (step 18): status=Status.UNKNOWN, low=0.0000000, high=0.0000019, mid=0.0000019, abs_max=1024.72607421875
rel_dist={0: [-886.6835943626116, 886.6835943634572]}

## Binary search (step 19) starts
Candidate diff: 0.0000010


## IAR start
Binary search (step 19): status=Status.UNKNOWN, low=0.0000000, high=0.0000010, mid=0.0000010, abs_max=1024.72607421875
rel_dist={0: [-886.6835943127753, 886.6835943184742]}

## Binary Search Result
Binary search time: 79.91 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 1116.37 seconds

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6885743, upper bound: 886.6823830
time: 0.88 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6793348, upper bound: 886.6793348
time: 0.81 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.85 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.85
Output dim: 0, lower bound: -886.6885743, upper bound: 886.6823830
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.85
Output dim: 0, lower bound: -886.6793348, upper bound: 886.6793348

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -373.4723206, 606.2720947, -390.5541382, 634.1719971, -1007.6442871, 996.8262329
1: -418.3924866, 599.0228271, -437.5506897, 626.8630371, -1045.2554932, 1036.5734863
2: -419.9433899, 592.4771118, -439.2138672, 620.2371826, -1040.1805420, 1031.6909180
3: -513.4045410, 690.0853882, -536.9467773, 721.8648682, -1235.2694092, 1227.0322266
4: -453.0448608, 677.0018921, -473.3275146, 708.4596558, -1161.5045166, 1150.3293457

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6793348, upper bound: 886.6793348
time: 1.31 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6793348, upper bound: 886.6793348
time: 1.47 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -506.5198975, 819.3720703, -386.4144592, 627.2296143, -1133.7495117, 1205.7863770
1: -573.1215820, 818.7601929, -432.8936462, 620.2084961, -1193.3300781, 1251.6536865
2: -569.8142090, 807.9828491, -434.5277405, 613.5854492, -1183.3996582, 1242.5106201
3: -704.5859375, 940.3079834, -531.2470703, 714.2178345, -1418.8037109, 1471.5546875
4: -613.9079590, 924.3903198, -468.2237244, 700.8625488, -1314.7705078, 1392.6140137

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6793348, upper bound: 886.6793348
time: 0.90 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6793348, upper bound: 886.6793348
time: 0.91 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.69 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.69
Output dim: 0, lower bound: -886.6793348, upper bound: 886.6793348
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.69
Output dim: 0, lower bound: -886.6793348, upper bound: 886.6793348
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.69
Output dim: 0, lower bound: -886.6793348, upper bound: 886.6793348
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.69
Output dim: 0, lower bound: -886.6793348, upper bound: 886.6793348

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -373.4723206, 606.2720947, -373.4723206, 606.2720947, -979.7443848, 979.7443848
1: -418.3924866, 599.0228271, -418.3924866, 599.0228271, -1017.4151001, 1017.4151611
2: -419.9433899, 592.4771118, -419.9433899, 592.4771118, -1012.4205322, 1012.4204712
3: -513.4045410, 690.0853882, -513.4045410, 690.0853882, -1203.4899902, 1203.4899902
4: -453.0448608, 677.0018921, -453.0448608, 677.0018921, -1130.0467529, 1130.0467529

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6828663, upper bound: 886.6719679
time: 1.28 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6817577, upper bound: 886.6724708
time: 1.42 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -373.4723206, 606.2720947, -506.5198975, 819.3720703, -1192.8441162, 1112.7918701
1: -418.3924866, 599.0228271, -573.1215820, 818.7601929, -1237.1527100, 1172.1444092
2: -419.9433899, 592.4771118, -569.8142090, 807.9828491, -1227.9262695, 1162.2911377
3: -513.4045410, 690.0853882, -704.5859375, 940.3079834, -1453.7124023, 1394.6711426
4: -453.0448608, 677.0018921, -613.9079590, 924.3903198, -1377.4351807, 1290.9099121

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6828663, upper bound: 886.6719679
time: 1.10 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6817577, upper bound: 886.6724708
time: 0.82 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -506.5198975, 819.3720703, -373.4723206, 606.2720947, -1112.7919922, 1192.8442383
1: -573.1215820, 818.7601929, -418.3924866, 599.0228271, -1172.1444092, 1237.1527100
2: -569.8142090, 807.9828491, -419.9433899, 592.4771118, -1162.2911377, 1227.9262695
3: -704.5859375, 940.3079834, -513.4045410, 690.0853882, -1394.6711426, 1453.7124023
4: -613.9079590, 924.3903198, -453.0448608, 677.0018921, -1290.9097900, 1377.4351807

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6740730, upper bound: 886.6693100
time: 0.93 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6694226, upper bound: 886.6694226
time: 1.18 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -506.5198975, 819.3720703, -506.5198975, 819.3720703, -1325.8917236, 1325.8918457
1: -573.1215820, 818.7601929, -573.1215820, 818.7601929, -1391.2940674, 1391.2940674
2: -569.8142090, 807.9828491, -569.8142090, 807.9828491, -1377.6153564, 1377.6153564
3: -704.5859375, 940.3079834, -704.5859375, 940.3079834, -1644.8936768, 1644.8936768
4: -613.9079590, 924.3903198, -613.9079590, 924.3903198, -1536.3483887, 1536.3483887

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6740730, upper bound: 886.6693100
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6694226, upper bound: 886.6694226
time: 1.30 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.15 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.15
Output dim: 0, lower bound: -886.6828663, upper bound: 886.6719679
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.15
Output dim: 0, lower bound: -886.6817577, upper bound: 886.6724708
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.15
Output dim: 0, lower bound: -886.6828663, upper bound: 886.6719679
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.15
Output dim: 0, lower bound: -886.6817577, upper bound: 886.6724708
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.15
Output dim: 0, lower bound: -886.6740730, upper bound: 886.6693100
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.15
Output dim: 0, lower bound: -886.6694226, upper bound: 886.6694226
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.15
Output dim: 0, lower bound: -886.6740730, upper bound: 886.6693100
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.15
Output dim: 0, lower bound: -886.6694226, upper bound: 886.6694226

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -349.6350098, 569.3693848, -371.1477356, 602.5512695, -952.1862793, 940.5170898
1: -392.1717224, 561.8006592, -415.8064270, 595.2962646, -987.4677734, 977.6070557
2: -393.7890320, 555.6325073, -417.3486633, 588.7767334, -982.5657959, 972.9812012
3: -480.6766052, 646.9263306, -510.1958313, 685.7918701, -1166.4681396, 1157.1217041
4: -425.8119507, 634.2166138, -450.3166809, 672.7362671, -1098.5479736, 1084.5333252

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6858484, upper bound: 886.6839486
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6858402, upper bound: 886.6830644
time: 1.19 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -364.6583252, 591.8895874, -373.4723206, 606.2720947, -970.9304199, 965.3618774
1: -408.4007263, 584.5307007, -418.3924866, 599.0228271, -1007.4234619, 1002.9232178
2: -410.0544434, 578.2492065, -419.9433899, 592.4771118, -1002.5314941, 998.1925659
3: -501.2101135, 673.4092407, -513.4045410, 690.0853882, -1191.2955322, 1186.8137207
4: -442.3909302, 660.6585083, -453.0448608, 677.0018921, -1119.3928223, 1113.7033691

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6839218, upper bound: 886.6847978
time: 1.05 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6839136, upper bound: 886.6839136
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -349.6350098, 569.3693848, -504.5165710, 816.1098022, -1165.7448730, 1073.8856201
1: -392.1717224, 561.8006592, -570.8804321, 815.4421387, -1207.6137695, 1132.6809082
2: -393.7890320, 555.6325073, -567.5739136, 804.7042236, -1198.4932861, 1123.2064209
3: -480.6766052, 646.9263306, -701.8140869, 936.5130615, -1417.1896973, 1348.7403564
4: -425.8119507, 634.2166138, -611.5197144, 920.6674194, -1346.4791260, 1245.7363281

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6828002, upper bound: 886.6715763
time: 1.43 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6816451, upper bound: 886.6719679
time: 1.13 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6816451, upper bound: 886.6719679
time: 1.10 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -364.6583252, 591.8895874, -506.5198975, 819.3720703, -1184.0302734, 1098.4094238
1: -408.4007263, 584.5307007, -573.1215820, 818.7601929, -1227.1608887, 1157.6523438
2: -410.0544434, 578.2492065, -569.8142090, 807.9828491, -1218.0373535, 1148.0634766
3: -501.2101135, 673.4092407, -704.5859375, 940.3079834, -1441.5179443, 1377.9951172
4: -442.3909302, 660.6585083, -613.9079590, 924.3903198, -1366.7812500, 1274.5664062

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6808736, upper bound: 886.6724255
time: 1.15 seconds

## Relational analysis of IS_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6816451, upper bound: 886.6724708
time: 1.15 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6816451, upper bound: 886.6724708
time: 1.36 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -482.6397095, 780.5917358, -371.1477356, 602.5512695, -1085.1909180, 1151.7395020
1: -546.5246582, 779.3952637, -415.8064270, 595.2962646, -1141.8208008, 1195.2016602
2: -543.2442627, 769.0665283, -417.3486633, 588.7767334, -1132.0209961, 1186.4151611
3: -671.4299927, 895.2281494, -510.1958313, 685.7918701, -1357.2218018, 1405.4238281
4: -586.0380249, 880.0111694, -450.3166809, 672.7362671, -1258.7739258, 1330.3278809

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6719679, upper bound: 886.6816451
time: 0.96 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6719679, upper bound: 886.6816451
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -492.6635437, 797.7097778, -373.4723206, 606.2720947, -1098.9356689, 1171.1818848
1: -557.5224609, 796.6813965, -418.3924866, 599.0228271, -1156.5452881, 1215.0738525
2: -554.4656372, 786.1628418, -419.9433899, 592.4771118, -1146.9426270, 1206.1059570
3: -685.3289185, 915.2006226, -513.4045410, 690.0853882, -1375.4143066, 1428.6052246
4: -597.6687012, 899.1670532, -453.0448608, 677.0018921, -1274.6705322, 1352.2119141

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6719679, upper bound: 886.6817577
time: 1.13 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6719679, upper bound: 886.6817577
time: 1.15 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -482.6397095, 780.5917358, -504.5165710, 816.1098022, -1298.7495117, 1285.1082764
1: -546.5246582, 779.3952637, -570.8804321, 815.4421387, -1361.2904053, 1349.4089355
2: -543.2442627, 769.0665283, -567.5739136, 804.7042236, -1347.6616211, 1336.2746582
3: -671.4299927, 895.2281494, -701.8140869, 936.5130615, -1607.7691650, 1596.7149658
4: -586.0380249, 880.0111694, -611.5197144, 920.6674194, -1504.8214111, 1489.6241455

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6693100, upper bound: 886.6693100
time: 1.09 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6693100, upper bound: 886.6693100
time: 1.24 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -492.6635437, 797.7097778, -506.5198975, 819.3720703, -1312.0355225, 1304.2294922
1: -557.5224609, 796.6813965, -573.1215820, 818.7601929, -1375.7152100, 1369.1549072
2: -554.4656372, 786.1628418, -569.8142090, 807.9828491, -1362.2453613, 1355.7015381
3: -685.3289185, 915.2006226, -704.5859375, 940.3079834, -1625.6369629, 1619.7866211
4: -597.6687012, 899.1670532, -613.9079590, 924.3903198, -1520.0410156, 1511.1147461

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6693100, upper bound: 886.6694226
time: 1.10 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6693100, upper bound: 886.6694226
time: 0.95 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.26 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 0, lower bound: -886.6858484, upper bound: 886.6839486
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 0, lower bound: -886.6858402, upper bound: 886.6830644
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 0, lower bound: -886.6839218, upper bound: 886.6847978
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 0, lower bound: -886.6839136, upper bound: 886.6839136
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 0, lower bound: -886.6816451, upper bound: 886.6719679
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 0, lower bound: -886.6816451, upper bound: 886.6719679
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 0, lower bound: -886.6816451, upper bound: 886.6724708
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 0, lower bound: -886.6816451, upper bound: 886.6724708
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 0, lower bound: -886.6719679, upper bound: 886.6816451
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 0, lower bound: -886.6719679, upper bound: 886.6816451
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 0, lower bound: -886.6719679, upper bound: 886.6817577
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 0, lower bound: -886.6719679, upper bound: 886.6817577
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 0, lower bound: -886.6693100, upper bound: 886.6693100
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 0, lower bound: -886.6693100, upper bound: 886.6693100
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 0, lower bound: -886.6693100, upper bound: 886.6694226
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.26
Output dim: 0, lower bound: -886.6693100, upper bound: 886.6694226

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -349.6350098, 569.3693848, -331.1970825, 530.6585693, -880.2935791, 900.5664062
1: -392.1717224, 561.8006592, -370.6009216, 526.0095825, -918.1812744, 932.4016113
2: -393.7890320, 555.6325073, -372.2444458, 520.7279663, -914.5169678, 927.8769531
3: -480.6766052, 646.9263306, -454.4379578, 605.7380371, -1086.4141846, 1101.3642578
4: -425.8119507, 634.2166138, -401.2303467, 595.8291016, -1021.6410522, 1035.4470215

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6858402, upper bound: 886.6830644
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6858402, upper bound: 886.6830644
time: 1.36 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -348.3105164, 567.3392944, -509.5909729, 826.6294556, -1174.9399414, 1076.9299316
1: -390.7347717, 559.7677002, -574.0014038, 825.6984863, -1216.4332275, 1133.7687988
2: -392.3096008, 553.5932617, -572.8388062, 816.0645752, -1208.3741455, 1126.4318848
3: -478.8839722, 644.5926514, -706.3557129, 946.6676025, -1425.5515137, 1350.9483643
4: -424.2928162, 631.9147949, -615.9163208, 933.9309082, -1357.1860352, 1247.8310547

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6858402, upper bound: 886.6830644
time: 1.10 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6858402, upper bound: 886.6830644
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -364.6583252, 591.8895874, -333.5416260, 534.4788208, -899.1371460, 925.4310913
1: -408.4007263, 584.5307007, -373.2019348, 529.8374634, -938.2381592, 957.7326660
2: -410.0544434, 578.2492065, -374.8469849, 524.5579834, -934.6124268, 953.0961914
3: -501.2101135, 673.4092407, -457.6570740, 610.2064819, -1111.4165039, 1131.0662842
4: -442.3909302, 660.6585083, -403.9684753, 600.1394043, -1042.5302734, 1064.6269531

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6839136, upper bound: 886.6839136
time: 1.15 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6839136, upper bound: 886.6839136
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -362.4306946, 588.2139282, -511.1507263, 829.1965942, -1191.6273193, 1099.3646240
1: -405.9606628, 580.8535767, -575.7471313, 828.2751465, -1234.2358398, 1156.6007080
2: -407.5406494, 574.5543213, -574.5769653, 818.6187744, -1226.1593018, 1149.1312256
3: -498.1868591, 669.1885376, -708.5311890, 949.6296387, -1447.8165283, 1377.7197266
4: -439.7738647, 656.5881958, -617.7683105, 936.8413086, -1375.5877686, 1274.3564453

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6839136, upper bound: 886.6839136
time: 1.09 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6839136, upper bound: 886.6839136
time: 1.08 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -349.6350098, 569.3693848, -482.6397095, 780.5917358, -1130.2268066, 1052.0087891
1: -392.1717224, 561.8006592, -546.5246582, 779.3952637, -1171.5668945, 1108.3249512
2: -393.7890320, 555.6325073, -543.2442627, 769.0665283, -1162.8555908, 1098.8767090
3: -480.6766052, 646.9263306, -671.4299927, 895.2281494, -1375.9047852, 1318.3562012
4: -425.8119507, 634.2166138, -586.0380249, 880.0111694, -1305.8231201, 1220.2546387

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6813062, upper bound: 886.6700152
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6733506, upper bound: 886.6693686
time: 1.66 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -349.6350098, 569.3693848, -492.6635437, 797.7097778, -1147.3446045, 1062.0328369
1: -392.1717224, 561.8006592, -557.5224609, 796.6813965, -1188.8529053, 1119.3231201
2: -393.7890320, 555.6325073, -554.4656372, 786.1628418, -1179.9519043, 1110.0980225
3: -480.6766052, 646.9263306, -685.3289185, 915.2006226, -1395.8769531, 1332.2552490
4: -425.8119507, 634.2166138, -597.6687012, 899.1670532, -1324.9790039, 1231.8852539

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6813062, upper bound: 886.6700152
time: 1.43 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6733506, upper bound: 886.6693686
time: 1.50 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -364.6583252, 591.8895874, -482.6397095, 780.5917358, -1145.2500000, 1074.5290527
1: -408.4007263, 584.5307007, -546.5246582, 779.3952637, -1187.7957764, 1131.0554199
2: -410.0544434, 578.2492065, -543.2442627, 769.0665283, -1179.1209717, 1121.4934082
3: -501.2101135, 673.4092407, -671.4299927, 895.2281494, -1396.4382324, 1344.8392334
4: -442.3909302, 660.6585083, -586.0380249, 880.0111694, -1322.4020996, 1246.6965332

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6657274, upper bound: 886.6698702
time: 1.10 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6813428, upper bound: 886.6724708
time: 1.17 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -364.6583252, 591.8895874, -492.6635437, 797.7097778, -1162.3680420, 1084.5531006
1: -408.4007263, 584.5307007, -557.5224609, 796.6813965, -1205.0820312, 1142.0532227
2: -410.0544434, 578.2492065, -554.4656372, 786.1628418, -1196.2171631, 1132.7148438
3: -501.2101135, 673.4092407, -685.3289185, 915.2006226, -1416.4107666, 1358.7381592
4: -442.3909302, 660.6585083, -597.6687012, 899.1670532, -1341.5579834, 1258.3271484

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6657274, upper bound: 886.6698702
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6813428, upper bound: 886.6724708
time: 1.45 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -482.6397095, 780.5917358, -349.6350098, 569.3693848, -1052.0089111, 1130.2268066
1: -546.5246582, 779.3952637, -392.1717224, 561.8006592, -1108.3250732, 1171.5667725
2: -543.2442627, 769.0665283, -393.7890320, 555.6325073, -1098.8767090, 1162.8555908
3: -671.4299927, 895.2281494, -480.6766052, 646.9263306, -1318.3562012, 1375.9047852
4: -586.0380249, 880.0111694, -425.8119507, 634.2166138, -1220.2546387, 1305.8231201

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6762639, upper bound: 886.6807609
time: 1.02 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6560203, upper bound: 886.6771718
time: 1.72 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6766184, upper bound: 886.6814194
time: 1.31 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -482.6397095, 780.5917358, -364.6583252, 591.8895874, -1074.5290527, 1145.2500000
1: -546.5246582, 779.3952637, -408.4007263, 584.5307007, -1131.0552979, 1187.7957764
2: -543.2442627, 769.0665283, -410.0544434, 578.2492065, -1121.4934082, 1179.1209717
3: -671.4299927, 895.2281494, -501.2101135, 673.4092407, -1344.8392334, 1396.4382324
4: -586.0380249, 880.0111694, -442.3909302, 660.6585083, -1246.6965332, 1322.4020996

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6762639, upper bound: 886.6807609
time: 1.09 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6560203, upper bound: 886.6771718
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6766184, upper bound: 886.6814194
time: 1.65 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -492.6635437, 797.7097778, -349.6350098, 569.3693848, -1062.0328369, 1147.3446045
1: -557.5224609, 796.6813965, -392.1717224, 561.8006592, -1119.3231201, 1188.8529053
2: -554.4656372, 786.1628418, -393.7890320, 555.6325073, -1110.0979004, 1179.9519043
3: -685.3289185, 915.2006226, -480.6766052, 646.9263306, -1332.2552490, 1395.8769531
4: -597.6687012, 899.1670532, -425.8119507, 634.2166138, -1231.8852539, 1324.9790039

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6715763, upper bound: 886.6808736
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6644527, upper bound: 886.6743173
time: 1.00 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6658429, upper bound: 886.6807594
time: 1.24 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6719679, upper bound: 886.6813428
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -492.6635437, 797.7097778, -364.6583252, 591.8895874, -1084.5531006, 1162.3680420
1: -557.5224609, 796.6813965, -408.4007263, 584.5307007, -1142.0532227, 1205.0819092
2: -554.4656372, 786.1628418, -410.0544434, 578.2492065, -1132.7148438, 1196.2170410
3: -685.3289185, 915.2006226, -501.2101135, 673.4092407, -1358.7381592, 1416.4107666
4: -597.6687012, 899.1670532, -442.3909302, 660.6585083, -1258.3271484, 1341.5579834

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6715763, upper bound: 886.6808736
time: 0.97 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6644527, upper bound: 886.6744478
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6658429, upper bound: 886.6807594
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6719679, upper bound: 886.6813428
time: 1.03 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -482.6397095, 780.5917358, -482.6397095, 780.5917358, -1263.2313232, 1263.2313232
1: -546.5246582, 779.3952637, -546.5246582, 779.3952637, -1325.0220947, 1325.0220947
2: -543.2442627, 769.0665283, -543.2442627, 769.0665283, -1311.8763428, 1311.8762207
3: -671.4299927, 895.2281494, -671.4299927, 895.2281494, -1566.1594238, 1566.1594238
4: -586.0380249, 880.0111694, -586.0380249, 880.0111694, -1464.1713867, 1464.1712646

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6530600, upper bound: 886.6648367
time: 1.14 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6736581, upper bound: 886.6690843
time: 1.16 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -482.6397095, 780.5917358, -492.6635437, 797.7097778, -1280.3491211, 1273.2552490
1: -546.5246582, 779.3952637, -557.5224609, 796.6813965, -1342.5037842, 1336.0732422
2: -543.2442627, 769.0665283, -554.4656372, 786.1628418, -1329.0621338, 1323.1424561
3: -671.4299927, 895.2281494, -685.3289185, 915.2006226, -1586.4588623, 1580.2938232
4: -586.0380249, 880.0111694, -597.6687012, 899.1670532, -1483.2961426, 1475.6826172

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6530600, upper bound: 886.6648367
time: 1.29 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6736581, upper bound: 886.6690843
time: 1.10 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -492.6635437, 797.7097778, -482.6397095, 780.5917358, -1273.2552490, 1280.3491211
1: -557.5224609, 796.6813965, -546.5246582, 779.3952637, -1336.0732422, 1342.5037842
2: -554.4656372, 786.1628418, -543.2442627, 769.0665283, -1323.1424561, 1329.0620117
3: -685.3289185, 915.2006226, -671.4299927, 895.2281494, -1580.2938232, 1586.4587402
4: -597.6687012, 899.1670532, -586.0380249, 880.0111694, -1475.6828613, 1483.2961426

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6628826, upper bound: 886.6684243
time: 1.10 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6690077, upper bound: 886.6690077
time: 1.18 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -492.6635437, 797.7097778, -492.6635437, 797.7097778, -1290.3731689, 1290.3732910
1: -557.5224609, 796.6813965, -557.5224609, 796.6813965, -1353.5760498, 1353.5760498
2: -554.4656372, 786.1628418, -554.4656372, 786.1628418, -1340.3315430, 1340.3315430
3: -685.3289185, 915.2006226, -685.3289185, 915.2006226, -1600.5295410, 1600.5295410
4: -597.6687012, 899.1670532, -597.6687012, 899.1670532, -1494.8073730, 1494.8073730

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6628826, upper bound: 886.6684243
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6690077, upper bound: 886.6690077
time: 1.01 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.94 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 0, lower bound: -886.6858402, upper bound: 886.6830644
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 0, lower bound: -886.6858402, upper bound: 886.6830644
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 0, lower bound: -886.6858402, upper bound: 886.6830644
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 0, lower bound: -886.6858402, upper bound: 886.6830644
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 0, lower bound: -886.6839136, upper bound: 886.6839136
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 0, lower bound: -886.6839136, upper bound: 886.6839136
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 0, lower bound: -886.6839136, upper bound: 886.6839136
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 0, lower bound: -886.6839136, upper bound: 886.6839136
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 0, lower bound: -886.6813062, upper bound: 886.6700152
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 0, lower bound: -886.6733506, upper bound: 886.6693686
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 0, lower bound: -886.6813062, upper bound: 886.6700152
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 0, lower bound: -886.6733506, upper bound: 886.6693686
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 0, lower bound: -886.6657274, upper bound: 886.6698702
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 0, lower bound: -886.6813428, upper bound: 886.6724708
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 0, lower bound: -886.6657274, upper bound: 886.6698702
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 0, lower bound: -886.6813428, upper bound: 886.6724708
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 0, lower bound: -886.6560203, upper bound: 886.6771718
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 0, lower bound: -886.6766184, upper bound: 886.6814194
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 0, lower bound: -886.6560203, upper bound: 886.6771718
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 0, lower bound: -886.6766184, upper bound: 886.6814194
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 0, lower bound: -886.6658429, upper bound: 886.6807594
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 0, lower bound: -886.6719679, upper bound: 886.6813428
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 0, lower bound: -886.6658429, upper bound: 886.6807594
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 0, lower bound: -886.6719679, upper bound: 886.6813428
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 0, lower bound: -886.6530600, upper bound: 886.6648367
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 0, lower bound: -886.6736581, upper bound: 886.6690843
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 0, lower bound: -886.6530600, upper bound: 886.6648367
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 0, lower bound: -886.6736581, upper bound: 886.6690843
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 0, lower bound: -886.6628826, upper bound: 886.6684243
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 0, lower bound: -886.6690077, upper bound: 886.6690077
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 0, lower bound: -886.6628826, upper bound: 886.6684243
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 0, lower bound: -886.6690077, upper bound: 886.6690077

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -309.7154236, 497.5668945, -331.1970825, 530.6585693, -840.3740234, 828.7637939
1: -346.9890747, 492.0843201, -370.6009216, 526.0095825, -872.9986572, 862.6852417
2: -348.7414856, 487.0979004, -372.2444458, 520.7279663, -869.4694824, 859.3423462
3: -425.0345459, 566.6265869, -454.4379578, 605.7380371, -1030.7723389, 1021.0645752
4: -376.9709473, 557.5439453, -401.2303467, 595.8291016, -972.8000488, 958.7739868

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6832181, upper bound: 886.6683218
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6858484, upper bound: 886.6839486
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -491.5893860, 797.4088135, -331.1970825, 530.6585693, -1022.2478638, 1128.6058350
1: -553.8572998, 796.1446533, -370.6009216, 526.0095825, -1079.8669434, 1166.7454834
2: -552.8887329, 786.7453003, -372.2444458, 520.7279663, -1073.6165771, 1158.9897461
3: -681.1791992, 912.6603394, -454.4379578, 605.7380371, -1286.9169922, 1367.0981445
4: -594.9200439, 900.4223022, -401.2303467, 595.8291016, -1190.7490234, 1300.9064941

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6832181, upper bound: 886.6683218
time: 1.29 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6858484, upper bound: 886.6839486
time: 1.10 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -309.7154236, 497.5668945, -509.5909729, 826.6294556, -1136.3448486, 1007.1578369
1: -346.9890747, 492.0843201, -574.0014038, 825.6984863, -1172.6873779, 1066.0854492
2: -348.7414856, 487.0979004, -572.8388062, 816.0645752, -1164.8060303, 1059.9366455
3: -425.0345459, 566.6265869, -706.3557129, 946.6676025, -1371.7021484, 1272.9822998
4: -376.9709473, 557.5439453, -615.9163208, 933.9309082, -1310.1647949, 1173.4600830

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6797689, upper bound: 886.6372055
time: 1.25 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6858402, upper bound: 886.6830644
time: 1.62 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -491.5893860, 797.4088135, -509.5909729, 826.6294556, -1318.2187500, 1306.9996338
1: -553.8572998, 796.1446533, -574.0014038, 825.6984863, -1377.9337158, 1368.5433350
2: -552.8887329, 786.7453003, -572.8388062, 816.0645752, -1367.2656250, 1357.9653320
3: -681.1791992, 912.6603394, -706.3557129, 946.6676025, -1626.9847412, 1618.0595703
4: -594.9200439, 900.4223022, -615.9163208, 933.9309082, -1526.1228027, 1513.6809082

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6797689, upper bound: 886.6372055
time: 1.17 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6858402, upper bound: 886.6830644
time: 1.12 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -324.6807556, 519.9534912, -333.5416260, 534.4788208, -859.1595459, 853.4949951
1: -363.1039124, 515.1547241, -373.2019348, 529.8374634, -892.9414062, 888.3566895
2: -364.9215393, 510.3396912, -374.8469849, 524.5579834, -889.4794922, 885.1866455
3: -445.3571777, 593.4895020, -457.6570740, 610.2064819, -1055.5635986, 1051.1462402
4: -393.2788391, 583.7371826, -403.9684753, 600.1394043, -993.4182129, 987.7056885

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6812915, upper bound: 886.6691817
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6839218, upper bound: 886.6847978
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -492.3429565, 798.1012573, -333.5416260, 534.4788208, -1026.8217773, 1131.6428223
1: -554.5442505, 796.8940430, -373.2019348, 529.8374634, -1084.3814697, 1170.0959473
2: -553.4963379, 787.6232910, -374.8469849, 524.5579834, -1078.0543213, 1162.4699707
3: -682.4190674, 913.6898193, -457.6570740, 610.2064819, -1292.6254883, 1371.3468018
4: -595.0189209, 901.7127686, -403.9684753, 600.1394043, -1195.1583252, 1305.0334473

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6812915, upper bound: 886.6691817
time: 1.17 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6839218, upper bound: 886.6847978
time: 1.13 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -324.6807556, 519.9534912, -511.1507263, 829.1965942, -1153.8773193, 1031.1042480
1: -363.1039124, 515.1547241, -575.7471313, 828.2751465, -1191.3790283, 1090.9018555
2: -364.9215393, 510.3396912, -574.5769653, 818.6187744, -1183.5401611, 1084.9166260
3: -445.3571777, 593.4895020, -708.5311890, 949.6296387, -1394.9868164, 1302.0206299
4: -393.2788391, 583.7371826, -617.7683105, 936.8413086, -1329.2049561, 1201.5053711

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6778701, upper bound: 886.6380547
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6839136, upper bound: 886.6839136
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -492.3429565, 798.1012573, -511.1507263, 829.1965942, -1321.5394287, 1309.2519531
1: -554.5442505, 796.8940430, -575.7471313, 828.2751465, -1381.4162598, 1371.3209229
2: -553.4963379, 787.6232910, -574.5769653, 818.6187744, -1370.5588379, 1360.8834229
3: -682.4190674, 913.6898193, -708.5311890, 949.6296387, -1631.5036621, 1621.6135254
4: -595.0189209, 901.7127686, -617.7683105, 936.8413086, -1529.0943604, 1516.9223633

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6778701, upper bound: 886.6380547
time: 1.47 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6839136, upper bound: 886.6839136
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -341.6400757, 556.9874878, -482.6397095, 780.5917358, -1122.2318115, 1039.6271973
1: -383.3443298, 549.1125488, -546.5246582, 779.3952637, -1162.7395020, 1095.6368408
2: -384.7740479, 543.1148682, -543.2442627, 769.0665283, -1153.8405762, 1086.3591309
3: -469.8532104, 632.5046387, -671.4299927, 895.2281494, -1365.0812988, 1303.9345703
4: -416.3528137, 619.5456543, -586.0380249, 880.0111694, -1296.3640137, 1205.5836182

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6813062, upper bound: 886.6739492
time: 1.21 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6765177, upper bound: 886.6555478
time: 1.11 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6806609, upper bound: 886.6742166
time: 1.16 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -358.2708740, 587.8734131, -480.3938599, 777.1262207, -1135.3970947, 1068.2670898
1: -401.0378418, 577.9248047, -543.9367065, 775.7813110, -1176.8189697, 1121.8612061
2: -403.3467102, 571.8195190, -540.7058716, 765.4598389, -1168.8065186, 1112.5253906
3: -491.6117859, 666.0189209, -668.2336426, 891.1484985, -1382.7602539, 1334.2521973
4: -434.1039429, 651.0877686, -583.1933594, 875.8226318, -1309.9265137, 1234.2811279

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6685621, upper bound: 886.6548933
time: 1.07 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6727052, upper bound: 886.6735621
time: 1.85 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -341.6400757, 556.9874878, -492.6635437, 797.7097778, -1139.3497314, 1049.6510010
1: -383.3443298, 549.1125488, -557.5224609, 796.6813965, -1180.0257568, 1106.6350098
2: -384.7740479, 543.1148682, -554.4656372, 786.1628418, -1170.9367676, 1097.5804443
3: -469.8532104, 632.5046387, -685.3289185, 915.2006226, -1385.0538330, 1317.8334961
4: -416.3528137, 619.5456543, -597.6687012, 899.1670532, -1315.5198975, 1217.2142334

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6814652, upper bound: 886.6697477
time: 1.39 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6710538, upper bound: 886.6616420
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6797342, upper bound: 886.6654558
time: 1.36 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6806217, upper bound: 886.6700152
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -358.2708740, 587.8734131, -490.3520508, 794.1398315, -1152.4105225, 1078.2254639
1: -401.0378418, 577.9248047, -554.8618774, 792.9476929, -1193.9855957, 1132.7863770
2: -403.3467102, 571.8195190, -551.8532104, 782.4519043, -1185.7985840, 1123.6727295
3: -491.6117859, 666.0189209, -682.0415039, 911.0087891, -1402.6206055, 1348.0600586
4: -434.1039429, 651.0877686, -594.7468262, 894.8574829, -1328.9614258, 1245.8345947

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6717785, upper bound: 886.6648014
time: 1.33 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6726660, upper bound: 886.6693607
time: 1.34 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -335.8760071, 547.7655029, -480.2530823, 776.8106079, -1112.6864014, 1028.0185547
1: -376.9928589, 541.9160767, -543.8724976, 775.7017212, -1152.6944580, 1085.7884521
2: -377.7377014, 535.4390869, -540.5502930, 765.3847046, -1143.1224365, 1075.9892578
3: -463.1000977, 624.3510132, -668.2198486, 890.9804077, -1354.0804443, 1292.5705566
4: -407.6806335, 610.6179199, -583.1108398, 875.7814941, -1283.4621582, 1193.7285156

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6615564, upper bound: 886.6539226
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6615564, upper bound: 886.6745207
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -362.8229065, 588.9429932, -482.6397095, 780.5917358, -1143.4146729, 1071.5826416
1: -406.3486328, 581.6405640, -546.5246582, 779.3952637, -1185.7438965, 1128.1651611
2: -407.9841614, 575.3658447, -543.2442627, 769.0665283, -1177.0506592, 1118.6099854
3: -498.7056580, 670.1018677, -671.4299927, 895.2281494, -1393.9338379, 1341.5314941
4: -440.1969910, 657.3359375, -586.0380249, 880.0111694, -1320.2081299, 1243.3740234

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6805353, upper bound: 886.6770032
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6771718, upper bound: 886.6565232
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6771718, upper bound: 886.6771213
time: 1.14 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -335.8760071, 547.7655029, -490.2459106, 793.8734741, -1129.7492676, 1038.0111084
1: -376.9928589, 541.9160767, -554.8367920, 792.9451904, -1169.9379883, 1096.7529297
2: -377.7377014, 535.4390869, -551.7357788, 782.4256592, -1160.1633301, 1087.1746826
3: -463.1000977, 624.3510132, -682.0761719, 910.8918457, -1373.9915771, 1306.4271240
4: -407.6806335, 610.6179199, -594.7053223, 894.8835449, -1302.5642090, 1205.3232422

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6651440, upper bound: 886.6637452
time: 1.04 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6651440, upper bound: 886.6698702
time: 1.36 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -362.8229065, 588.9429932, -492.6635437, 797.7097778, -1160.5327148, 1081.6065674
1: -406.3486328, 581.6405640, -557.5224609, 796.6813965, -1203.0300293, 1139.1630859
2: -407.9841614, 575.3658447, -554.4656372, 786.1628418, -1194.1468506, 1129.8311768
3: -498.7056580, 670.1018677, -685.3289185, 915.2006226, -1413.9062500, 1355.4306641
4: -440.1969910, 657.3359375, -597.6687012, 899.1670532, -1339.3640137, 1255.0046387

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6804586, upper bound: 886.6724255
time: 1.26 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6738311, upper bound: 886.6645237
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6792782, upper bound: 886.6663458
time: 1.06 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6792782, upper bound: 886.6724708
time: 1.24 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -476.7803650, 771.6308594, -347.2109985, 565.4932251, -1042.2733154, 1118.8417969
1: -539.9763184, 772.4961548, -389.4451904, 558.0382690, -1098.0144043, 1161.9411621
2: -536.3087158, 761.8306885, -391.0382080, 551.8658447, -1088.1745605, 1152.8687744
3: -664.6658936, 886.8717041, -477.3734741, 642.5969238, -1307.2628174, 1364.2451172
4: -577.7337036, 871.2992554, -422.7757568, 629.8594971, -1207.5931396, 1294.0748291

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6555478, upper bound: 886.6765177
time: 1.43 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6548933, upper bound: 886.6685621
time: 1.04 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -480.2467346, 776.7702637, -349.6350098, 569.3693848, -1049.6160889, 1126.4051514
1: -543.8629150, 775.5766602, -392.1717224, 561.8006592, -1105.6634521, 1167.7481689
2: -540.5451050, 765.2640991, -393.7890320, 555.6325073, -1096.1774902, 1159.0531006
3: -668.1536255, 890.8625488, -480.6766052, 646.9263306, -1315.0799561, 1371.5389404
4: -583.1776733, 875.6425781, -425.8119507, 634.2166138, -1217.3942871, 1301.4543457

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6742166, upper bound: 886.6806609
time: 1.04 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6735621, upper bound: 886.6727052
time: 1.38 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -476.7803650, 771.6308594, -362.2507324, 588.0256348, -1064.8056641, 1133.8814697
1: -539.9763184, 772.4961548, -405.6886292, 580.7615356, -1120.7377930, 1178.1848145
2: -536.3087158, 761.8306885, -407.3251648, 574.4904175, -1110.7990723, 1169.1557617
3: -664.6658936, 886.8717041, -497.9349976, 669.0750732, -1333.7409668, 1384.8066406
4: -577.7337036, 871.2992554, -439.3639221, 656.3276367, -1234.0611572, 1310.6632080

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6539226, upper bound: 886.6615564
time: 0.90 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6539226, upper bound: 886.6771718
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -480.2467346, 776.7702637, -364.6583252, 591.8895874, -1072.1363525, 1141.4285889
1: -543.8629150, 775.5766602, -408.4007263, 584.5307007, -1128.3935547, 1183.9771729
2: -540.5451050, 765.2640991, -410.0544434, 578.2492065, -1118.7943115, 1175.3183594
3: -668.1536255, 890.8625488, -501.2101135, 673.4092407, -1341.5628662, 1392.0726318
4: -583.1776733, 875.6425781, -442.3909302, 660.6585083, -1243.8361816, 1318.0333252

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6745207, upper bound: 886.6658041
time: 1.35 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6745207, upper bound: 886.6814194
time: 0.98 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -483.8485718, 783.6398926, -347.2109985, 565.4932251, -1049.3417969, 1130.8508301
1: -547.6936646, 784.4992065, -389.4451904, 558.0382690, -1105.7319336, 1173.9440918
2: -544.0818481, 773.7381592, -391.0382080, 551.8658447, -1095.9477539, 1164.7762451
3: -674.3672485, 900.7269897, -477.3734741, 642.5969238, -1316.9641113, 1378.1004639
4: -585.6768188, 884.7250366, -422.7757568, 629.8594971, -1215.5363770, 1307.5006104

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6654558, upper bound: 886.6797342
time: 0.96 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6648014, upper bound: 886.6717785
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -490.4929810, 794.2290649, -349.6350098, 569.3693848, -1059.8620605, 1143.8640137
1: -555.0930786, 793.1872559, -392.1717224, 561.8006592, -1116.8936768, 1185.3588867
2: -552.0131836, 782.6972656, -393.7890320, 555.6325073, -1107.6455078, 1176.4863281
3: -682.3418579, 911.2104492, -480.6766052, 646.9263306, -1329.2681885, 1391.8869629
4: -595.0556030, 895.1871338, -425.8119507, 634.2166138, -1229.2722168, 1320.9990234

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6700152, upper bound: 886.6806217
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6693607, upper bound: 886.6726660
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -483.8485718, 783.6398926, -362.2507324, 588.0256348, -1071.8741455, 1145.8906250
1: -547.6936646, 784.4992065, -405.6886292, 580.7615356, -1128.4552002, 1190.1878662
2: -544.0818481, 773.7381592, -407.3251648, 574.4904175, -1118.5722656, 1181.0632324
3: -674.3672485, 900.7269897, -497.9349976, 669.0750732, -1343.4423828, 1398.6619873
4: -585.6768188, 884.7250366, -439.3639221, 656.3276367, -1242.0043945, 1324.0889893

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6630269, upper bound: 886.6618973
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6630269, upper bound: 886.6807594
time: 0.96 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -490.4929810, 794.2290649, -364.6583252, 591.8895874, -1082.3824463, 1158.8874512
1: -555.0930786, 793.1872559, -408.4007263, 584.5307007, -1139.6237793, 1201.5878906
2: -552.0131836, 782.6972656, -410.0544434, 578.2492065, -1130.2624512, 1192.7517090
3: -682.3418579, 911.2104492, -501.2101135, 673.4092407, -1355.7510986, 1412.4205322
4: -595.0556030, 895.1871338, -442.3909302, 660.6585083, -1255.7141113, 1337.5781250

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6646706, upper bound: 886.6618973
time: 1.00 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6646706, upper bound: 886.6813428
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -476.7803650, 771.6308594, -480.2530823, 776.8106079, -1253.5909424, 1251.8837891
1: -539.9763184, 772.4961548, -543.8724976, 775.7017212, -1314.3925781, 1314.4536133
2: -536.3087158, 761.8306885, -540.5502930, 765.3847046, -1300.9996338, 1300.7888184
3: -664.6658936, 886.8717041, -668.2198486, 890.9804077, -1554.6072998, 1553.3597412
4: -577.7337036, 871.2992554, -583.1108398, 875.7814941, -1451.2772217, 1451.8018799

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6488890, upper bound: 886.6488890
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6488890, upper bound: 886.6694871
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -480.2467346, 776.7702637, -482.6397095, 780.5917358, -1260.8385010, 1259.4096680
1: -543.8629150, 775.5766602, -546.5246582, 779.3952637, -1322.2313232, 1320.9387207
2: -540.5451050, 765.2640991, -543.2442627, 769.0665283, -1309.0322266, 1307.7741699
3: -668.1536255, 890.8625488, -671.4299927, 895.2281494, -1562.6138916, 1561.4362793
4: -583.1776733, 875.6425781, -586.0380249, 880.0111694, -1461.2619629, 1459.6694336

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6670839, upper bound: 886.6489390
time: 1.23 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6670839, upper bound: 886.6737347
time: 0.99 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -476.7803650, 771.6308594, -490.2459106, 793.8734741, -1270.6538086, 1261.8767090
1: -539.9763184, 772.4961548, -554.8367920, 792.9451904, -1331.8364258, 1325.4720459
2: -536.3087158, 761.8306885, -551.7357788, 782.4256592, -1318.1400146, 1312.0194092
3: -664.6658936, 886.8717041, -682.0761719, 910.8918457, -1574.8510742, 1567.4523926
4: -577.7337036, 871.2992554, -594.7053223, 894.8835449, -1470.3553467, 1463.2828369

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6524766, upper bound: 886.6587116
time: 1.30 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6524766, upper bound: 886.6648367
time: 1.06 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -480.2467346, 776.7702637, -492.6635437, 797.7097778, -1277.9562988, 1269.4337158
1: -543.8629150, 775.5766602, -557.5224609, 796.6813965, -1339.7130127, 1331.9897461
2: -540.5451050, 765.2640991, -554.4656372, 786.1628418, -1326.2180176, 1319.0402832
3: -668.1536255, 890.8625488, -685.3289185, 915.2006226, -1582.9130859, 1575.5704346
4: -583.1776733, 875.6425781, -597.6687012, 899.1670532, -1480.3867188, 1471.1806641

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6730747, upper bound: 886.6629592
time: 1.00 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6730747, upper bound: 886.6690843
time: 1.66 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -483.8485718, 783.6398926, -480.2530823, 776.8106079, -1260.6591797, 1263.8929443
1: -547.6936646, 784.4992065, -543.8724976, 775.7017212, -1322.1848145, 1326.5643311
2: -544.0818481, 773.7381592, -540.5502930, 765.3847046, -1308.7917480, 1312.7916260
3: -674.3672485, 900.7269897, -668.2198486, 890.9804077, -1564.5362549, 1567.4094238
4: -585.6768188, 884.7250366, -583.1108398, 875.7814941, -1459.1884766, 1465.2464600

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6505756, upper bound: 886.6488890
time: 0.98 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6505756, upper bound: 886.6730747
time: 1.12 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -490.4929810, 794.2290649, -482.6397095, 780.5917358, -1271.0847168, 1276.8686523
1: -555.0930786, 793.1872559, -546.5246582, 779.3952637, -1333.5136719, 1338.6999512
2: -552.0131836, 782.6972656, -543.2442627, 769.0665283, -1320.5397949, 1325.2807617
3: -682.3418579, 911.2104492, -671.4299927, 895.2281494, -1577.0366211, 1582.0471191
4: -595.0556030, 895.1871338, -586.0380249, 880.0111694, -1473.0228271, 1479.1925049

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6525777, upper bound: 886.6488890
time: 1.14 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6525777, upper bound: 886.6736581
time: 1.53 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -483.8485718, 783.6398926, -490.2459106, 793.8734741, -1277.7220459, 1273.8857422
1: -547.6936646, 784.4992065, -554.8367920, 792.9451904, -1339.6552734, 1337.6103516
2: -544.0818481, 773.7381592, -551.7357788, 782.4256592, -1325.9361572, 1324.0340576
3: -674.3672485, 900.7269897, -682.0761719, 910.8918457, -1584.8372803, 1581.5572510
4: -585.6768188, 884.7250366, -594.7053223, 894.8835449, -1478.2664795, 1476.7274170

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6622992, upper bound: 886.6618290
time: 1.29 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6622992, upper bound: 886.6684243
time: 1.16 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -490.4929810, 794.2290649, -492.6635437, 797.7097778, -1288.2026367, 1286.8925781
1: -555.0930786, 793.1872559, -557.5224609, 796.6813965, -1351.0167236, 1349.7724609
2: -552.0131836, 782.6972656, -554.4656372, 786.1628418, -1337.7288818, 1336.5506592
3: -682.3418579, 911.2104492, -685.3289185, 915.2006226, -1597.3839111, 1596.2297363
4: -595.0556030, 895.1871338, -597.6687012, 899.1670532, -1492.1477051, 1490.7039795

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6646706, upper bound: 886.6618290
time: 0.99 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6646706, upper bound: 886.6690077
time: 1.12 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 5.37 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6832181, upper bound: 886.6683218
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6858484, upper bound: 886.6839486
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6832181, upper bound: 886.6683218
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6858484, upper bound: 886.6839486
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6797689, upper bound: 886.6372055
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6858402, upper bound: 886.6830644
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6797689, upper bound: 886.6372055
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6858402, upper bound: 886.6830644
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6812915, upper bound: 886.6691817
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6839218, upper bound: 886.6847978
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6812915, upper bound: 886.6691817
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6839218, upper bound: 886.6847978
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6778701, upper bound: 886.6380547
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6839136, upper bound: 886.6839136
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6778701, upper bound: 886.6380547
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6839136, upper bound: 886.6839136
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6765177, upper bound: 886.6555478
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6806609, upper bound: 886.6742166
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6685621, upper bound: 886.6548933
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6727052, upper bound: 886.6735621
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6797342, upper bound: 886.6654558
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6806217, upper bound: 886.6700152
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6717785, upper bound: 886.6648014
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6726660, upper bound: 886.6693607
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6615564, upper bound: 886.6539226
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6615564, upper bound: 886.6745207
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6771718, upper bound: 886.6565232
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6771718, upper bound: 886.6771213
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6651440, upper bound: 886.6637452
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6651440, upper bound: 886.6698702
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6792782, upper bound: 886.6663458
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6792782, upper bound: 886.6724708
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6555478, upper bound: 886.6765177
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6548933, upper bound: 886.6685621
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6742166, upper bound: 886.6806609
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6735621, upper bound: 886.6727052
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6539226, upper bound: 886.6615564
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6539226, upper bound: 886.6771718
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6745207, upper bound: 886.6658041
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6745207, upper bound: 886.6814194
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6654558, upper bound: 886.6797342
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6648014, upper bound: 886.6717785
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6700152, upper bound: 886.6806217
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6693607, upper bound: 886.6726660
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6630269, upper bound: 886.6618973
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6630269, upper bound: 886.6807594
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6646706, upper bound: 886.6618973
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6646706, upper bound: 886.6813428
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6488890, upper bound: 886.6488890
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6488890, upper bound: 886.6694871
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6670839, upper bound: 886.6489390
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6670839, upper bound: 886.6737347
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6524766, upper bound: 886.6587116
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6524766, upper bound: 886.6648367
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6730747, upper bound: 886.6629592
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6730747, upper bound: 886.6690843
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6505756, upper bound: 886.6488890
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6505756, upper bound: 886.6730747
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6525777, upper bound: 886.6488890
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6525777, upper bound: 886.6736581
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6622992, upper bound: 886.6618290
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6622992, upper bound: 886.6684243
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6646706, upper bound: 886.6618290
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 0, lower bound: -886.6646706, upper bound: 886.6690077

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -307.3939209, 493.9909058, -298.9349365, 480.0113525, -787.4052734, 792.9258423
1: -344.3673706, 488.4537354, -335.4577637, 476.7706299, -821.1380005, 823.9113770
2: -346.1005859, 483.4774780, -335.8315430, 471.1593933, -817.2600098, 819.3090210
3: -421.8644104, 562.5387573, -411.6441040, 548.9973755, -970.8618164, 974.1828613
4: -374.0190125, 553.3673096, -361.8763733, 538.9319458, -912.9509277, 915.2435913

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6801385, upper bound: 886.6683542
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6722938, upper bound: 886.6676997
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -309.7154236, 497.5668945, -328.6104126, 526.5114136, -836.2268066, 826.1771851
1: -346.9890747, 492.0843201, -367.6697083, 521.8521118, -868.8411865, 859.7539673
2: -348.7414856, 487.0979004, -369.3340454, 516.5917358, -865.3331299, 856.4318848
3: -425.0345459, 566.6265869, -450.8390808, 600.9822388, -1026.0167236, 1017.4656982
4: -376.9709473, 557.5439453, -398.1561890, 591.0501099, -968.0210571, 955.7000732

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6832285, upper bound: 886.6752629
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6754097, upper bound: 886.6746084
time: 1.23 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -489.8291626, 794.7731323, -298.9349365, 480.0113525, -969.8405151, 1093.7080078
1: -551.9039307, 793.5413818, -335.4577637, 476.7706299, -1028.6745605, 1128.9991455
2: -550.8955688, 784.1384277, -335.8315430, 471.1593933, -1022.0549316, 1119.9699707
3: -678.8648071, 909.6679077, -411.6441040, 548.9973755, -1227.8620605, 1321.3120117
4: -592.7493286, 897.4163818, -361.8763733, 538.9319458, -1131.6812744, 1258.5465088

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6801385, upper bound: 886.6680867
time: 1.45 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6832181, upper bound: 886.6665028
time: 1.27 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6831834, upper bound: 886.6673484
time: 1.10 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -491.5893860, 797.4088135, -328.6104126, 526.5114136, -1018.1007690, 1126.0192871
1: -553.8572998, 796.1446533, -367.6697083, 521.8521118, -1075.7092285, 1163.8140869
2: -552.8887329, 786.7453003, -369.3340454, 516.5917358, -1069.4804688, 1156.0793457
3: -681.1791992, 912.6603394, -450.8390808, 600.9822388, -1282.1611328, 1363.4991455
4: -594.9200439, 900.4223022, -398.1561890, 591.0501099, -1185.9702148, 1297.8259277

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6832285, upper bound: 886.6749954
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6855137, upper bound: 886.6821298
time: 1.31 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6854790, upper bound: 886.6829752
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -307.3939209, 493.9909058, -484.4414062, 788.4041748, -1095.7980957, 978.4323120
1: -344.3673706, 488.4537354, -546.4085083, 788.0035400, -1132.3708496, 1034.8623047
2: -346.1005859, 483.4774780, -544.6184082, 778.3785400, -1124.4788818, 1028.0957031
3: -421.8644104, 562.5387573, -672.7867432, 903.2171021, -1325.0812988, 1235.3254395
4: -374.0190125, 553.3673096, -585.9557495, 890.6555176, -1263.7447510, 1139.3229980

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6674175, upper bound: 886.6160801
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6662481, upper bound: 886.6158239
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -309.7154236, 497.5668945, -505.6334534, 820.0872803, -1129.8027344, 1003.2003174
1: -346.9890747, 492.0843201, -569.5653076, 819.3248901, -1166.3137207, 1061.6496582
2: -348.7414856, 487.0979004, -568.3519897, 809.7286377, -1158.4700928, 1055.4499512
3: -425.0345459, 566.6265869, -700.9290771, 939.3248901, -1364.3593750, 1267.5556641
4: -376.9709473, 557.5439453, -611.0194092, 926.6738281, -1302.8607178, 1168.5632324

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6790289, upper bound: 886.6768033
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6773389, upper bound: 886.6765152
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -489.8291626, 794.7731323, -484.4414062, 788.4041748, -1278.2333984, 1279.2144775
1: -551.9039307, 793.5413818, -546.4085083, 788.0035400, -1338.1699219, 1338.3244629
2: -550.8955688, 784.1384277, -544.6184082, 778.3785400, -1327.4699707, 1327.0863037
3: -678.8648071, 909.6679077, -672.7867432, 903.2171021, -1581.2713623, 1581.4918213
4: -592.7493286, 897.4163818, -585.9557495, 890.6555176, -1480.4880371, 1480.5668945

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6797689, upper bound: 886.6353867
time: 1.22 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6797487, upper bound: 886.6362321
time: 1.19 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -491.5893860, 797.4088135, -505.6334534, 820.0872803, -1311.6766357, 1303.0422363
1: -553.8572998, 796.1446533, -569.5653076, 819.3248901, -1371.4588623, 1363.9978027
2: -552.8887329, 786.7453003, -568.3519897, 809.7286377, -1360.7622070, 1353.3588867
3: -681.1791992, 912.6603394, -700.9290771, 939.3248901, -1619.5373535, 1612.4298096
4: -594.9200439, 900.4223022, -611.0194092, 926.6738281, -1518.8189697, 1508.8436279

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6855137, upper bound: 886.6812456
time: 1.14 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6854790, upper bound: 886.6820910
time: 1.34 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -322.3567505, 516.3554688, -301.2014771, 483.7593994, -806.1161499, 817.5568848
1: -360.4837646, 511.5952148, -337.9712219, 480.4739685, -840.9577637, 849.5663452
2: -362.2780457, 506.7789917, -338.3539734, 474.8296204, -837.1076660, 845.1329346
3: -442.1914978, 589.4053345, -414.7583313, 553.3012695, -995.4927979, 1004.1636963
4: -390.3316650, 579.5601196, -364.5399475, 543.0929565, -933.4246216, 944.0999146

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6665608, upper bound: 886.6665608
time: 1.19 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6665608, upper bound: 886.6691900
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -324.6807556, 519.9534912, -330.9510193, 530.3215942, -855.0023193, 850.9044800
1: -363.1039124, 515.1547241, -370.2655640, 525.6689453, -888.7727661, 885.4202881
2: -364.9215393, 510.3396912, -371.9326172, 520.3974609, -885.3189697, 882.2722778
3: -445.3571777, 593.4895020, -454.0519714, 605.4384155, -1050.7956543, 1047.5413818
4: -393.2788391, 583.7371826, -400.8913879, 595.3486328, -988.6274414, 984.6284790

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6691900, upper bound: 886.6821762
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6691900, upper bound: 886.6848060
time: 1.53 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -490.5940247, 795.4718018, -301.2014771, 483.7593994, -974.3533936, 1096.6732178
1: -552.6021118, 794.3042603, -337.9712219, 480.4739685, -1033.0760498, 1132.2755127
2: -551.5175781, 785.0286255, -338.3539734, 474.8296204, -1026.3471680, 1123.3823242
3: -680.1116943, 910.7183228, -414.7583313, 553.3012695, -1233.4128418, 1325.4766846
4: -592.8713989, 898.7221069, -364.5399475, 543.0929565, -1135.9643555, 1262.6313477

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6785059, upper bound: 886.6689455
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6354328, upper bound: 886.6631300
time: 1.35 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6354328, upper bound: 886.6691817
time: 1.35 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -492.3429565, 798.1012573, -330.9510193, 530.3215942, -1022.6645508, 1129.0522461
1: -554.5442505, 796.8940430, -370.2655640, 525.6689453, -1080.2131348, 1167.1596680
2: -553.4963379, 787.6232910, -371.9326172, 520.3974609, -1073.8937988, 1159.5557861
3: -682.4190674, 913.6898193, -454.0519714, 605.4384155, -1287.8574219, 1367.7418213
4: -595.0189209, 901.7127686, -400.8913879, 595.3486328, -1190.3674316, 1301.9530029

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6380629, upper bound: 886.6787542
time: 1.28 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6380629, upper bound: 886.6847978
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -322.3567505, 516.3554688, -485.9776306, 790.9172974, -1113.2740479, 1002.3331299
1: -360.4837646, 511.5952148, -548.1273804, 790.5266724, -1151.0102539, 1059.7224121
2: -362.2780457, 506.7789917, -546.3283081, 780.8836060, -1143.1614990, 1053.1072998
3: -442.1914978, 589.4053345, -674.9259644, 906.1143799, -1348.3057861, 1264.3310547
4: -390.3316650, 579.5601196, -587.7765503, 893.5131226, -1282.7562256, 1167.3365479

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6631300, upper bound: 886.6354328
time: 1.08 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6631300, upper bound: 886.6380629
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -324.6807556, 519.9534912, -507.1879883, 822.6466064, -1147.3273926, 1027.1413574
1: -363.1039124, 515.1547241, -571.3057251, 821.8931274, -1184.9970703, 1086.4604492
2: -364.9215393, 510.3396912, -570.0839844, 812.2729492, -1177.1944580, 1080.4234619
3: -445.3571777, 593.4895020, -703.0986328, 942.2771606, -1387.6342773, 1296.5877686
4: -393.2788391, 583.7371826, -612.8648071, 929.5744629, -1321.8896484, 1196.6020508

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6691816, upper bound: 886.6812915
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6691816, upper bound: 886.6839218
time: 1.11 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -490.5940247, 795.4718018, -485.9776306, 790.9172974, -1281.5113525, 1281.4494629
1: -552.6021118, 794.3042603, -548.1273804, 790.5266724, -1341.6140137, 1341.0928955
2: -551.5175781, 785.0286255, -546.3283081, 780.8836060, -1330.7321777, 1329.9937744
3: -680.1116943, 910.7183228, -674.9259644, 906.1143799, -1585.7392578, 1585.0366211
4: -592.8713989, 898.7221069, -587.7765503, 893.5131226, -1483.4354248, 1483.7998047

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6320111, upper bound: 886.6320112
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6320111, upper bound: 886.6380547
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -492.3429565, 798.1012573, -507.1879883, 822.6466064, -1314.9895020, 1305.2893066
1: -554.5442505, 796.8940430, -571.3057251, 821.8931274, -1374.9312744, 1366.7689209
2: -553.4963379, 787.6232910, -570.0839844, 812.2729492, -1364.0440674, 1356.2701416
3: -682.4190674, 913.6898193, -703.0986328, 942.2771606, -1624.0451660, 1615.9765625
4: -595.0189209, 901.7127686, -612.8648071, 929.5744629, -1521.7790527, 1512.0780029

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6380547, upper bound: 886.6778701
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6380547, upper bound: 886.6839136
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -339.2185059, 553.1197510, -476.7803650, 771.6308594, -1110.8491211, 1029.8999023
1: -380.6207886, 545.3540649, -539.9763184, 772.4961548, -1153.1169434, 1085.3303223
2: -382.0084839, 539.3532104, -536.3087158, 761.8306885, -1143.8388672, 1075.6618652
3: -466.5535583, 628.1792603, -664.6658936, 886.8717041, -1353.4250488, 1292.8450928
4: -413.3217773, 615.1891479, -577.7337036, 871.2992554, -1284.1337891, 1192.9226074

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6723467, upper bound: 886.6479430
time: 1.18 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6703657, upper bound: 886.6452317
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -341.6400757, 556.9874878, -480.2467346, 776.7702637, -1118.4101562, 1037.2342529
1: -383.3443298, 549.1125488, -543.8629150, 775.5766602, -1158.9208984, 1092.9753418
2: -384.7740479, 543.1148682, -540.5451050, 765.2640991, -1150.0379639, 1083.6599121
3: -469.8532104, 632.5046387, -668.1536255, 890.8625488, -1360.7158203, 1300.6582031
4: -416.3528137, 619.5456543, -583.1776733, 875.6425781, -1291.9953613, 1202.7232666

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6802557, upper bound: 886.6742166
time: 1.27 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6782747, upper bound: 886.6715054
time: 1.08 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -355.7970581, 584.0119019, -474.5835571, 768.2429199, -1124.0400391, 1058.5954590
1: -398.2730103, 574.1501465, -537.4403076, 768.9416504, -1167.2145996, 1111.5902100
2: -400.5617981, 568.0474243, -533.8213501, 758.2973022, -1158.8590088, 1101.8686523
3: -488.2570190, 661.6696167, -661.5311279, 882.8779297, -1371.1350098, 1323.2006836
4: -431.0721741, 646.7226562, -574.9531860, 867.1901245, -1298.2623291, 1221.6757812

Time for backsubstitution: 2.02 seconds
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=1024.72607421875
rel_dist={0: [-886.6916286331143, 886.6916286331141]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6841461, upper bound: 886.6805354
time: 0.81 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6765901, upper bound: 886.6765901
time: 0.87 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.84 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.84
Output dim: 0, lower bound: -886.6841461, upper bound: 886.6805354
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.84
Output dim: 0, lower bound: -886.6765901, upper bound: 886.6765901

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -373.4723206, 606.2720947, -387.1684265, 628.6696167, -1002.1419678, 993.4405518
1: -418.3924866, 599.0228271, -433.7556152, 621.3712769, -1039.7637939, 1032.7784424
2: -419.9433899, 592.4771118, -435.3907471, 614.7651978, -1034.7084961, 1027.8679199
3: -513.4045410, 690.0853882, -532.2945557, 715.5960083, -1229.0004883, 1222.3798828
4: -453.0448608, 677.0018921, -469.3018188, 702.2413330, -1155.2860107, 1146.3037109

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6765901, upper bound: 886.6765901
time: 0.92 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6765901, upper bound: 886.6765901
time: 1.07 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -506.5198975, 819.3720703, -376.4355164, 610.6185913, -1117.1384277, 1195.8074951
1: -573.1215820, 818.7601929, -421.6658630, 604.1460571, -1177.2675781, 1240.4259033
2: -569.8142090, 807.9828491, -423.2252197, 597.7636719, -1167.5778809, 1231.2080078
3: -704.5859375, 940.3079834, -517.5474854, 696.0693970, -1400.6551514, 1457.8553467
4: -613.9079590, 924.3903198, -455.8475952, 682.6322021, -1296.5401611, 1380.2379150

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6765901, upper bound: 886.6765901
time: 1.18 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6765901, upper bound: 886.6765901
time: 1.02 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.32 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.32
Output dim: 0, lower bound: -886.6765901, upper bound: 886.6765901
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.32
Output dim: 0, lower bound: -886.6765901, upper bound: 886.6765901
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.32
Output dim: 0, lower bound: -886.6765901, upper bound: 886.6765901
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.32
Output dim: 0, lower bound: -886.6765901, upper bound: 886.6765901

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -373.4723206, 606.2720947, -373.4723206, 606.2720947, -979.7443848, 979.7443848
1: -418.3924866, 599.0228271, -418.3924866, 599.0228271, -1017.4151001, 1017.4151611
2: -419.9433899, 592.4771118, -419.9433899, 592.4771118, -1012.4205322, 1012.4204712
3: -513.4045410, 690.0853882, -513.4045410, 690.0853882, -1203.4899902, 1203.4899902
4: -453.0448608, 677.0018921, -453.0448608, 677.0018921, -1130.0467529, 1130.0467529

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6772535, upper bound: 886.6713300
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6779965, upper bound: 886.6716574
time: 0.77 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -373.4723206, 606.2720947, -506.5198975, 819.3720703, -1192.8441162, 1112.7918701
1: -418.3924866, 599.0228271, -573.1215820, 818.7601929, -1237.1527100, 1172.1444092
2: -419.9433899, 592.4771118, -569.8142090, 807.9828491, -1227.9262695, 1162.2911377
3: -513.4045410, 690.0853882, -704.5859375, 940.3079834, -1453.7124023, 1394.6711426
4: -453.0448608, 677.0018921, -613.9079590, 924.3903198, -1377.4351807, 1290.9099121

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6772535, upper bound: 886.6713300
time: 1.38 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6779965, upper bound: 886.6716574
time: 0.90 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -506.5198975, 819.3720703, -373.4723206, 606.2720947, -1112.7919922, 1192.8442383
1: -573.1215820, 818.7601929, -418.3924866, 599.0228271, -1172.1444092, 1237.1527100
2: -569.8142090, 807.9828491, -419.9433899, 592.4771118, -1162.2911377, 1227.9262695
3: -704.5859375, 940.3079834, -513.4045410, 690.0853882, -1394.6711426, 1453.7124023
4: -613.9079590, 924.3903198, -453.0448608, 677.0018921, -1290.9097900, 1377.4351807

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6717866, upper bound: 886.6671811
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6671919, upper bound: 886.6671919
time: 0.79 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -506.5198975, 819.3720703, -506.5198975, 819.3720703, -1325.8917236, 1325.8918457
1: -573.1215820, 818.7601929, -573.1215820, 818.7601929, -1391.2940674, 1391.2940674
2: -569.8142090, 807.9828491, -569.8142090, 807.9828491, -1377.6153564, 1377.6153564
3: -704.5859375, 940.3079834, -704.5859375, 940.3079834, -1644.8936768, 1644.8936768
4: -613.9079590, 924.3903198, -613.9079590, 924.3903198, -1536.3483887, 1536.3483887

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6717866, upper bound: 886.6671811
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6671919, upper bound: 886.6671919
time: 0.79 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.70 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.70
Output dim: 0, lower bound: -886.6772535, upper bound: 886.6713300
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.70
Output dim: 0, lower bound: -886.6779965, upper bound: 886.6716574
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.70
Output dim: 0, lower bound: -886.6772535, upper bound: 886.6713300
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.70
Output dim: 0, lower bound: -886.6779965, upper bound: 886.6716574
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.70
Output dim: 0, lower bound: -886.6717866, upper bound: 886.6671811
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.70
Output dim: 0, lower bound: -886.6671919, upper bound: 886.6671919
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.70
Output dim: 0, lower bound: -886.6717866, upper bound: 886.6671811
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.70
Output dim: 0, lower bound: -886.6671919, upper bound: 886.6671919

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -349.6350098, 569.3693848, -360.6823425, 585.8190918, -935.4541016, 930.0517578
1: -392.1717224, 561.8006592, -404.1502991, 578.5072021, -970.6786499, 965.9509277
2: -393.7890320, 555.6325073, -405.6689453, 572.1100464, -965.8989868, 961.3014526
3: -480.6766052, 646.9263306, -495.7288513, 666.4656372, -1147.1418457, 1142.6551514
4: -425.8119507, 634.2166138, -438.0463562, 653.5093994, -1079.3212891, 1072.2629395

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6850519, upper bound: 886.6830959
time: 1.26 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6850519, upper bound: 886.6822169
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -364.6583252, 591.8895874, -370.5598450, 601.5141602, -966.1724854, 962.4494629
1: -408.4007263, 584.5307007, -415.0951233, 594.2302246, -1002.6309204, 999.6258545
2: -410.0544434, 578.2492065, -416.6785889, 587.7620850, -997.8164673, 994.9277954
3: -501.2101135, 673.4092407, -509.3659973, 684.5682373, -1185.7783203, 1182.7752686
4: -442.3909302, 660.6585083, -449.5257263, 671.5762939, -1113.9667969, 1110.1842041

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6831418, upper bound: 886.6840579
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6831418, upper bound: 886.6831418
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -349.6350098, 569.3693848, -495.1062927, 800.7984619, -1150.4333496, 1064.4754639
1: -392.1717224, 561.8006592, -560.3423462, 799.8886719, -1192.0604248, 1122.1428223
2: -393.7890320, 555.6325073, -557.0460205, 789.3265381, -1183.1156006, 1112.6783447
3: -480.6766052, 646.9263306, -688.7771606, 918.7344971, -1399.4105225, 1335.7032471
4: -425.8119507, 634.2166138, -600.2980347, 903.1672363, -1328.9792480, 1234.5146484

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6745333, upper bound: 886.6646690
time: 1.10 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6770733, upper bound: 886.6713300
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -364.6583252, 591.8895874, -502.8234253, 813.5553589, -1178.2136230, 1094.7130127
1: -408.4007263, 584.5307007, -568.9594727, 812.8254395, -1221.2259521, 1153.4901123
2: -410.0544434, 578.2492065, -565.7199707, 802.1174927, -1212.1718750, 1143.9692383
3: -501.2101135, 673.4092407, -699.4484253, 933.5614014, -1434.7714844, 1372.8576660
4: -442.3909302, 660.6585083, -609.5604858, 917.6389771, -1360.0296631, 1270.2189941

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6765829, upper bound: 886.6659847
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6777221, upper bound: 886.6716574
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -482.6397095, 780.5917358, -360.6823425, 585.8190918, -1068.4587402, 1141.2740479
1: -546.5246582, 779.3952637, -404.1502991, 578.5072021, -1125.0314941, 1183.5455322
2: -543.2442627, 769.0665283, -405.6689453, 572.1100464, -1115.3542480, 1174.7354736
3: -671.4299927, 895.2281494, -495.7288513, 666.4656372, -1337.8955078, 1390.9570312
4: -586.0380249, 880.0111694, -438.0463562, 653.5093994, -1239.5473633, 1318.0574951

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6718815, upper bound: 886.6643911
time: 0.97 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6752935, upper bound: 886.6778932
time: 1.22 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -492.6635437, 797.7097778, -370.5598450, 601.5141602, -1094.1777344, 1168.2695312
1: -557.5224609, 796.6813965, -415.0951233, 594.2302246, -1151.7525635, 1211.7763672
2: -554.4656372, 786.1628418, -416.6785889, 587.7620850, -1142.2277832, 1202.8414307
3: -685.3289185, 915.2006226, -509.3659973, 684.5682373, -1369.8972168, 1424.5666504
4: -597.6687012, 899.1670532, -449.5257263, 671.5762939, -1269.2446289, 1348.6927490

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6680516, upper bound: 886.6643818
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6716574, upper bound: 886.6777221
time: 0.97 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -482.6397095, 780.5917358, -495.1062927, 800.7984619, -1283.4377441, 1275.6979980
1: -546.5246582, 779.3952637, -560.3423462, 799.8886719, -1345.6654053, 1338.8267822
2: -543.2442627, 769.0665283, -557.0460205, 789.3265381, -1332.1966553, 1325.7581787
3: -671.4299927, 895.2281494, -688.7771606, 918.7344971, -1589.8538818, 1583.5714111
4: -586.0380249, 880.0111694, -600.2980347, 903.1672363, -1487.3953857, 1478.5125732

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6707582, upper bound: 886.6613470
time: 1.44 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6714147, upper bound: 886.6668871
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -492.6635437, 797.7097778, -502.8234253, 813.5553589, -1306.2188721, 1300.5332031
1: -557.5224609, 796.6813965, -568.9594727, 812.8254395, -1369.7556152, 1365.0012207
2: -554.4656372, 786.1628418, -565.7199707, 802.1174927, -1356.3505859, 1351.6051025
3: -685.3289185, 915.2006226, -699.4484253, 933.5614014, -1618.8903809, 1614.6490479
4: -597.6687012, 899.1670532, -609.5604858, 917.6389771, -1513.2921143, 1506.7446289

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6662007, upper bound: 886.6613022
time: 0.97 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6668636, upper bound: 886.6668636
time: 0.85 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 5.13 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.13
Output dim: 0, lower bound: -886.6850519, upper bound: 886.6830959
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.13
Output dim: 0, lower bound: -886.6850519, upper bound: 886.6822169
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.13
Output dim: 0, lower bound: -886.6831418, upper bound: 886.6840579
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.13
Output dim: 0, lower bound: -886.6831418, upper bound: 886.6831418
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.13
Output dim: 0, lower bound: -886.6745333, upper bound: 886.6646690
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.13
Output dim: 0, lower bound: -886.6770733, upper bound: 886.6713300
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.13
Output dim: 0, lower bound: -886.6765829, upper bound: 886.6659847
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.13
Output dim: 0, lower bound: -886.6777221, upper bound: 886.6716574
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.13
Output dim: 0, lower bound: -886.6718815, upper bound: 886.6643911
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.13
Output dim: 0, lower bound: -886.6752935, upper bound: 886.6778932
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.13
Output dim: 0, lower bound: -886.6680516, upper bound: 886.6643818
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.13
Output dim: 0, lower bound: -886.6716574, upper bound: 886.6777221
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.13
Output dim: 0, lower bound: -886.6707582, upper bound: 886.6613470
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.13
Output dim: 0, lower bound: -886.6714147, upper bound: 886.6668871
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.13
Output dim: 0, lower bound: -886.6662007, upper bound: 886.6613022
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.13
Output dim: 0, lower bound: -886.6668636, upper bound: 886.6668636

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -339.4937134, 550.5329590, -320.8171387, 513.9625244, -853.4562378, 871.3500977
1: -380.7462158, 543.6462402, -359.0826721, 509.2458191, -889.9920654, 902.7288818
2: -382.3465576, 537.6790161, -360.7201843, 504.0891724, -886.4356079, 898.3991699
3: -466.5447693, 625.7813110, -440.1816101, 586.2059326, -1052.7506104, 1065.9628906
4: -413.3014526, 614.5394287, -389.1264343, 576.9399414, -990.2413330, 1003.6658936

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6850519, upper bound: 886.6822169
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6850519, upper bound: 886.6822169
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -344.7338257, 561.7985840, -502.2540894, 814.5479126, -1159.2817383, 1064.0526123
1: -386.8819275, 554.2285156, -565.7893677, 813.5722656, -1200.4539795, 1120.0178223
2: -388.3000793, 548.0335693, -564.6612549, 804.0467529, -1192.3466797, 1112.6948242
3: -474.0536499, 638.2237549, -696.1177979, 932.7298584, -1406.7834473, 1334.3415527
4: -420.2182007, 625.6618042, -607.2116699, 920.2410889, -1339.4011230, 1232.8735352

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6850519, upper bound: 886.6822169
time: 1.54 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6850519, upper bound: 886.6822169
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -355.5996399, 575.0870972, -330.9267578, 530.1464844, -885.7460938, 906.0138550
1: -398.1553650, 568.3562012, -370.2261047, 525.4412231, -923.5965576, 938.5822754
2: -399.8100891, 562.5876465, -371.9234009, 520.3214111, -920.1314697, 934.5109863
3: -488.5918274, 654.6378784, -454.0174561, 605.2406616, -1093.8325195, 1108.6552734
4: -431.2857971, 643.1138306, -400.8283081, 595.2706299, -1026.5563965, 1043.9421387

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6831418, upper bound: 886.6831418
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6831418, upper bound: 886.6831418
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -357.7567749, 580.7260132, -506.3040771, 821.1828003, -1178.9394531, 1087.0299072
1: -400.9042053, 573.3439941, -570.2858887, 820.1856079, -1221.0894775, 1143.6298828
2: -402.3002014, 567.0504150, -569.1453857, 810.6233521, -1212.9234619, 1136.1958008
3: -491.8998718, 660.5205688, -701.8025513, 940.3638916, -1432.2634277, 1362.3231201
4: -434.3764648, 648.1927490, -611.9082031, 927.7840576, -1361.2592773, 1260.1009521

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6831418, upper bound: 886.6831418
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6831418, upper bound: 886.6831418
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -340.8757324, 555.3827515, -487.9498291, 789.5354614, -1130.4111328, 1043.3325195
1: -382.3272095, 548.2073975, -552.3477173, 790.6480103, -1172.9752197, 1100.5550537
2: -383.8684387, 542.0347290, -548.6082764, 779.7769775, -1163.6453857, 1090.6429443
3: -468.7704773, 631.2799072, -680.1488647, 907.6591797, -1376.4294434, 1311.4287109
4: -414.8437500, 618.5184937, -590.4536743, 891.9060669, -1306.7493896, 1208.9721680

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6725994, upper bound: 886.6641107
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6699064, upper bound: 886.6623718
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -349.3238525, 568.8692017, -492.9746704, 797.4037476, -1146.7275391, 1061.8438721
1: -391.8218384, 561.3052979, -557.9567261, 796.4893799, -1188.3112793, 1119.2619629
2: -393.4382324, 555.1408081, -554.6371460, 785.9524536, -1179.3905029, 1109.7777100
3: -480.2507629, 646.3619385, -685.8466187, 914.8530884, -1395.1036377, 1332.2082520
4: -425.4427795, 633.6483765, -597.7373047, 899.2791138, -1324.7218018, 1231.3857422

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6757273, upper bound: 886.6695775
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6706530, upper bound: 886.6670779
time: 1.12 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -355.8877258, 577.8049927, -494.8212280, 800.8665771, -1156.7541504, 1072.6262207
1: -398.5220947, 570.7971191, -560.0187988, 802.0792236, -1200.6010742, 1130.8157959
2: -400.1136169, 564.5828857, -556.2733765, 791.0935059, -1191.2070312, 1120.8560791
3: -489.2827454, 657.6176147, -689.6235352, 920.7554932, -1410.0378418, 1347.2409668
4: -431.3812866, 644.9000244, -598.6124878, 904.7331543, -1336.1143799, 1243.5124512

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6749005, upper bound: 886.6656453
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6690237, upper bound: 886.6624370
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -364.3976746, 591.4705200, -500.6353149, 810.0427246, -1174.4404297, 1092.1058350
1: -408.1093140, 584.1199341, -566.5096436, 809.3057861, -1217.4150391, 1150.6295166
2: -409.7603760, 577.8392944, -563.2463379, 798.6261597, -1208.3864746, 1141.0854492
3: -500.8544006, 672.9390259, -696.4406738, 929.5416260, -1430.3959961, 1369.3795166
4: -442.0795288, 660.1860962, -606.9240723, 913.6280518, -1355.7075195, 1267.1101074

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6643818, upper bound: 886.6680516
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6643818, upper bound: 886.6716574
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -474.2132263, 767.2434082, -334.1739502, 545.4487305, -1019.6617432, 1101.4173584
1: -537.1589355, 766.4085083, -375.3325806, 539.6484375, -1076.8073730, 1141.7410889
2: -533.7401733, 756.0997925, -375.9838562, 533.0645142, -1066.8045654, 1132.0834961
3: -660.0987549, 880.2410889, -460.8188782, 621.6968994, -1281.7956543, 1341.0599365
4: -575.7005005, 865.1007690, -406.2091675, 607.7661743, -1183.4663086, 1271.3099365

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6535926, upper bound: 886.6603363
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6535926, upper bound: 886.6643911
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -482.2932739, 780.0370483, -358.9269409, 583.0151367, -1065.3083496, 1138.9637451
1: -546.1387329, 778.8394775, -402.1892090, 575.7573853, -1121.8959961, 1181.0286865
2: -542.8527222, 768.5132446, -403.6869507, 569.3664551, -1112.2191162, 1172.1997070
3: -670.9551392, 894.5927734, -493.3361206, 663.3167114, -1334.2716064, 1387.9284668
4: -585.6217651, 879.3766479, -435.9633789, 650.3424072, -1235.9641113, 1315.3394775

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6563553, upper bound: 886.6728172
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6563553, upper bound: 886.6778932
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -483.8269958, 783.6589355, -342.6009827, 558.7576904, -1042.5845947, 1126.2598877
1: -547.7066040, 782.9964600, -384.6203613, 552.9998169, -1100.7062988, 1167.6168213
2: -544.4898071, 772.4794922, -385.3064575, 546.3264160, -1090.8161621, 1157.7854004
3: -673.4371948, 899.4168701, -472.4100342, 637.0861816, -1310.5234375, 1371.8269043
4: -586.8394775, 883.4949951, -415.8305969, 623.1256104, -1209.9650879, 1299.3255615

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6624819, upper bound: 886.6638296
time: 1.00 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6624819, upper bound: 886.6643818
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -492.3460693, 797.2020874, -368.7682800, 598.6347046, -1090.9807129, 1165.9703369
1: -557.1683960, 796.1719971, -413.0889893, 591.4052734, -1148.5737305, 1209.2609863
2: -554.1074829, 785.6572876, -414.6584473, 584.9446411, -1139.0518799, 1200.3156738
3: -684.8927002, 914.6182861, -506.9190063, 681.3372192, -1366.2299805, 1421.5372314
4: -597.2881470, 898.5863647, -447.3838501, 668.3334961, -1265.6215820, 1345.9702148

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6659847, upper bound: 886.6765829
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6659847, upper bound: 886.6777221
time: 0.98 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -474.2132263, 767.2434082, -487.9498291, 789.5354614, -1263.7486572, 1255.1932373
1: -537.1589355, 766.4085083, -552.3477173, 790.6480103, -1325.9111328, 1317.5245361
2: -533.7401733, 756.0997925, -548.6082764, 779.7769775, -1311.9833984, 1304.0759277
3: -660.0987549, 880.2410889, -680.1488647, 907.6591797, -1566.1248779, 1559.4620361
4: -575.7005005, 865.1007690, -590.4536743, 891.9060669, -1465.0257568, 1453.3596191

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6523250, upper bound: 886.6577162
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6523250, upper bound: 886.6613470
time: 1.12 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -482.2932739, 780.0370483, -492.9746704, 797.4037476, -1279.6968994, 1273.0117188
1: -546.1387329, 778.8394775, -557.9567261, 796.4893799, -1341.5560303, 1335.7075195
2: -542.8527222, 768.5132446, -554.6371460, 785.9524536, -1328.0950928, 1322.5942383
3: -670.9551392, 894.5927734, -685.8466187, 914.8530884, -1585.0404053, 1579.6684570
4: -585.6217651, 879.3766479, -597.7373047, 899.2791138, -1482.9592285, 1475.2435303

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6529107, upper bound: 886.6632870
time: 0.97 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6529107, upper bound: 886.6668871
time: 1.19 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -483.8269958, 783.6589355, -494.8212280, 800.8665771, -1284.6932373, 1278.4802246
1: -547.7066040, 782.9964600, -560.0187988, 802.0792236, -1348.0853271, 1342.0833740
2: -544.4898071, 772.4794922, -556.2733765, 791.0935059, -1334.2198486, 1328.2512207
3: -673.4371948, 899.4168701, -689.6235352, 920.7554932, -1593.0047607, 1588.6489258
4: -586.8394775, 883.4949951, -598.6124878, 904.7331543, -1488.8446045, 1479.8482666

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6607334, upper bound: 886.6607334
time: 0.98 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6607334, upper bound: 886.6613022
time: 0.95 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -492.3460693, 797.2020874, -500.6353149, 810.0427246, -1302.3883057, 1297.8372803
1: -557.1683960, 796.1719971, -566.5096436, 809.3057861, -1365.5480957, 1361.8531494
2: -554.1074829, 785.6572876, -563.2463379, 798.6261597, -1352.1551514, 1348.4161377
3: -684.8927002, 914.6182861, -696.4406738, 929.5416260, -1614.0887451, 1610.7772217
4: -597.2881470, 898.5863647, -606.9240723, 913.6280518, -1508.7598877, 1503.4562988

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6613022, upper bound: 886.6662007
time: 1.03 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6613022, upper bound: 886.6668636
time: 0.98 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 5.81 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.81
Output dim: 0, lower bound: -886.6850519, upper bound: 886.6822169
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.81
Output dim: 0, lower bound: -886.6850519, upper bound: 886.6822169
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.81
Output dim: 0, lower bound: -886.6850519, upper bound: 886.6822169
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.81
Output dim: 0, lower bound: -886.6850519, upper bound: 886.6822169
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.81
Output dim: 0, lower bound: -886.6831418, upper bound: 886.6831418
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.81
Output dim: 0, lower bound: -886.6831418, upper bound: 886.6831418
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.81
Output dim: 0, lower bound: -886.6831418, upper bound: 886.6831418
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.81
Output dim: 0, lower bound: -886.6831418, upper bound: 886.6831418
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.81
Output dim: 0, lower bound: -886.6725994, upper bound: 886.6641107
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.81
Output dim: 0, lower bound: -886.6699064, upper bound: 886.6623718
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.81
Output dim: 0, lower bound: -886.6757273, upper bound: 886.6695775
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.81
Output dim: 0, lower bound: -886.6706530, upper bound: 886.6670779
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.81
Output dim: 0, lower bound: -886.6749005, upper bound: 886.6656453
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.81
Output dim: 0, lower bound: -886.6690237, upper bound: 886.6624370
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.81
Output dim: 0, lower bound: -886.6643818, upper bound: 886.6680516
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.81
Output dim: 0, lower bound: -886.6643818, upper bound: 886.6716574
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.81
Output dim: 0, lower bound: -886.6535926, upper bound: 886.6603363
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.81
Output dim: 0, lower bound: -886.6535926, upper bound: 886.6643911
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.81
Output dim: 0, lower bound: -886.6563553, upper bound: 886.6728172
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.81
Output dim: 0, lower bound: -886.6563553, upper bound: 886.6778932
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.81
Output dim: 0, lower bound: -886.6624819, upper bound: 886.6638296
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.81
Output dim: 0, lower bound: -886.6624819, upper bound: 886.6643818
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.81
Output dim: 0, lower bound: -886.6659847, upper bound: 886.6765829
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.81
Output dim: 0, lower bound: -886.6659847, upper bound: 886.6777221
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.81
Output dim: 0, lower bound: -886.6523250, upper bound: 886.6577162
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.81
Output dim: 0, lower bound: -886.6523250, upper bound: 886.6613470
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.81
Output dim: 0, lower bound: -886.6529107, upper bound: 886.6632870
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.81
Output dim: 0, lower bound: -886.6529107, upper bound: 886.6668871
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.81
Output dim: 0, lower bound: -886.6607334, upper bound: 886.6607334
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.81
Output dim: 0, lower bound: -886.6607334, upper bound: 886.6613022
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.81
Output dim: 0, lower bound: -886.6613022, upper bound: 886.6662007
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.81
Output dim: 0, lower bound: -886.6613022, upper bound: 886.6668636

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -309.7154236, 497.5668945, -320.8171387, 513.9625244, -823.6779785, 818.3839722
1: -346.9890747, 492.0843201, -359.0826721, 509.2458191, -856.2348633, 851.1668701
2: -348.7414856, 487.0979004, -360.7201843, 504.0891724, -852.8306885, 847.8180542
3: -425.0345459, 566.6265869, -440.1816101, 586.2059326, -1011.2402954, 1006.8082275
4: -376.9709473, 557.5439453, -389.1264343, 576.9399414, -953.9108887, 946.6702881

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6764788, upper bound: 886.6791062
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6767064, upper bound: 886.6759886
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -491.5893860, 797.4088135, -320.8171387, 513.9625244, -1005.5518799, 1118.2259521
1: -553.8572998, 796.1446533, -359.0826721, 509.2458191, -1063.1030273, 1155.2271729
2: -552.8887329, 786.7453003, -360.7201843, 504.0891724, -1056.9779053, 1147.4654541
3: -681.1791992, 912.6603394, -440.1816101, 586.2059326, -1267.3850098, 1352.8414307
4: -594.9200439, 900.4223022, -389.1264343, 576.9399414, -1171.8599854, 1288.8414307

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6764788, upper bound: 886.6791062
time: 1.22 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6767064, upper bound: 886.6759886
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -309.7154236, 497.5668945, -502.2540894, 814.5479126, -1124.2633057, 999.8208008
1: -346.9890747, 492.0843201, -565.7893677, 813.5722656, -1160.5611572, 1057.8735352
2: -348.7414856, 487.0979004, -564.6612549, 804.0467529, -1152.7882080, 1051.7590332
3: -425.0345459, 566.6265869, -696.1177979, 932.7298584, -1357.7644043, 1262.7443848
4: -376.9709473, 557.5439453, -607.2116699, 920.2410889, -1296.4073486, 1164.7553711

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6841234, upper bound: 886.6822169
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6841234, upper bound: 886.6822169
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -491.5893860, 797.4088135, -502.2540894, 814.5479126, -1306.1373291, 1299.6628418
1: -553.8572998, 796.1446533, -565.7893677, 813.5722656, -1365.6127930, 1360.2669678
2: -552.8887329, 786.7453003, -564.6612549, 804.0467529, -1355.0611572, 1349.7523193
3: -681.1791992, 912.6603394, -696.1177979, 932.7298584, -1612.8095703, 1607.7169189
4: -594.9200439, 900.4223022, -607.2116699, 920.2410889, -1512.3654785, 1504.9700928

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6841234, upper bound: 886.6822169
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6841234, upper bound: 886.6822169
time: 1.18 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -324.6807556, 519.9534912, -330.9267578, 530.1464844, -854.8272705, 850.8802490
1: -363.1039124, 515.1547241, -370.2261047, 525.4412231, -888.5449829, 885.3808594
2: -364.9215393, 510.3396912, -371.9234009, 520.3214111, -885.2428589, 882.2629395
3: -445.3571777, 593.4895020, -454.0174561, 605.2406616, -1050.5979004, 1047.5069580
4: -393.2788391, 583.7371826, -400.8283081, 595.2706299, -988.5494385, 984.5654297

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6646718, upper bound: 886.6686843
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6831418, upper bound: 886.6840579
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -492.3429565, 798.1012573, -330.9267578, 530.1464844, -1022.4894409, 1129.0280762
1: -554.5442505, 796.8940430, -370.2261047, 525.4412231, -1079.9853516, 1167.1201172
2: -553.4963379, 787.6232910, -371.9234009, 520.3214111, -1073.8177490, 1159.5463867
3: -682.4190674, 913.6898193, -454.0174561, 605.2406616, -1287.6596680, 1367.7072754
4: -595.0189209, 901.7127686, -400.8283081, 595.2706299, -1190.2895508, 1301.9110107

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6646718, upper bound: 886.6686843
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6831418, upper bound: 886.6840579
time: 1.23 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -324.6807556, 519.9534912, -506.3040771, 821.1828003, -1145.8635254, 1026.2573242
1: -363.1039124, 515.1547241, -570.2858887, 820.1856079, -1183.2895508, 1085.4406738
2: -364.9215393, 510.3396912, -569.1453857, 810.6233521, -1175.5449219, 1079.4851074
3: -445.3571777, 593.4895020, -701.8025513, 940.3638916, -1385.7208252, 1295.2919922
4: -393.2788391, 583.7371826, -611.9082031, 927.7840576, -1320.2351074, 1195.6452637

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6610907, upper bound: 886.6378088
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6831418, upper bound: 886.6831418
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -492.3429565, 798.1012573, -506.3040771, 821.1828003, -1313.5255127, 1304.4052734
1: -554.5442505, 796.8940430, -570.2858887, 820.1856079, -1373.3923340, 1365.8946533
2: -553.4963379, 787.6232910, -569.1453857, 810.6233521, -1362.6514893, 1355.4686279
3: -682.4190674, 913.6898193, -701.8025513, 940.3638916, -1622.2871094, 1614.9379883
4: -595.0189209, 901.7127686, -611.9082031, 927.7840576, -1520.1243896, 1511.0952148

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6610907, upper bound: 886.6378088
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6831418, upper bound: 886.6831418
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -332.8414307, 542.9602051, -486.6910400, 787.5052490, -1120.3465576, 1029.6511230
1: -373.4511719, 535.4621582, -550.9429321, 788.5897217, -1162.0408936, 1086.3181152
2: -374.7404175, 529.4602051, -547.1805420, 777.7427979, -1152.4831543, 1076.6407471
3: -457.8814087, 616.7893066, -678.4187012, 905.3047485, -1363.1861572, 1295.2080078
4: -405.3461609, 603.7582397, -588.9039917, 889.5905762, -1293.9997559, 1192.5350342

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6596454, upper bound: 886.6374741
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6648529, upper bound: 886.6503966
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -349.2615662, 573.7863159, -476.2473145, 771.5094604, -1120.7708740, 1050.0335693
1: -390.9645691, 564.1651611, -538.8547974, 771.7719116, -1162.7364502, 1103.0197754
2: -393.2011719, 558.0673828, -535.3592529, 761.0133057, -1154.2144775, 1093.4266357
3: -479.3881531, 650.1647339, -663.4799805, 886.4668579, -1365.8548584, 1313.6445312
4: -423.0611572, 635.1975708, -575.7208862, 870.0654907, -1293.1267090, 1210.9183350

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6586119, upper bound: 886.6361233
time: 1.15 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6620789, upper bound: 886.6484595
time: 1.12 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -341.3329468, 556.4940796, -491.8562622, 795.5998535, -1136.9328613, 1048.3503418
1: -382.9986572, 548.6238403, -556.7066040, 794.6599731, -1177.6584473, 1105.2969971
2: -384.4276428, 542.6301880, -553.3631592, 784.1464844, -1168.5740967, 1095.9930420
3: -469.4330750, 631.9481201, -684.3110352, 912.7559204, -1382.1887207, 1316.2590332
4: -415.9869385, 618.9852905, -596.3532104, 897.2233887, -1312.9450684, 1215.3382568

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6757273, upper bound: 886.6695775
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6757273, upper bound: 886.6695775
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -357.9915466, 587.4293823, -481.0756226, 779.0277710, -1137.0191650, 1068.5050049
1: -400.7248230, 577.4871216, -544.2604980, 777.2798462, -1178.0046387, 1121.7475586
2: -403.0310974, 571.3848877, -541.1951904, 766.8622437, -1169.8933105, 1112.5800781
3: -491.2326660, 665.5183105, -668.9288330, 893.2967529, -1384.5294189, 1334.4467773
4: -433.7706909, 650.5820923, -582.7232666, 877.1063843, -1310.8768311, 1233.3054199

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6706530, upper bound: 886.6670779
time: 1.25 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6706530, upper bound: 886.6670779
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -348.7460022, 566.7811890, -493.5998230, 798.8869019, -1147.6329346, 1060.3808594
1: -390.6624756, 559.5158691, -558.6557617, 800.0761719, -1190.7386475, 1118.1716309
2: -392.0309753, 553.3423462, -554.8873901, 789.1110229, -1181.1419678, 1108.2294922
3: -479.5672913, 644.7905273, -687.9455566, 918.4648438, -1398.0319824, 1332.7359619
4: -422.8426514, 631.7910156, -597.1082764, 902.4790039, -1324.2939453, 1228.8009033

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6631920, upper bound: 886.6379143
time: 1.32 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6672226, upper bound: 886.6506555
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -362.4795227, 593.3336792, -483.1424255, 782.9064941, -1145.3859863, 1076.4760742
1: -405.1720581, 583.7545776, -546.5551147, 783.2534180, -1188.4252930, 1130.3096924
2: -407.3835449, 577.4948120, -543.0475464, 772.3878784, -1179.7712402, 1120.5423584
3: -497.2500610, 673.1428833, -672.9912109, 899.6297607, -1396.8798828, 1346.1340332
4: -437.1643066, 658.0410156, -583.9002075, 882.9561768, -1320.1204834, 1241.9411621

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6575481, upper bound: 886.6360001
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6600135, upper bound: 886.6483535
time: 1.22 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -335.8760071, 547.7655029, -500.6353149, 810.0427246, -1145.9183350, 1048.4005127
1: -376.9928589, 541.9160767, -566.5096436, 809.3057861, -1186.2984619, 1108.4257812
2: -377.7377014, 535.4390869, -563.2463379, 798.6261597, -1176.3637695, 1098.6850586
3: -463.1000977, 624.3510132, -696.4406738, 929.5416260, -1392.6416016, 1320.7913818
4: -407.6806335, 610.6179199, -606.9240723, 913.6280518, -1321.3087158, 1217.5419922

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6603363, upper bound: 886.6680516
time: 1.06 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6603363, upper bound: 886.6680516
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -362.8229065, 588.9429932, -500.6353149, 810.0427246, -1172.8653564, 1089.5783691
1: -406.3486328, 581.6405640, -566.5096436, 809.3057861, -1215.6544189, 1148.1501465
2: -407.9841614, 575.3658447, -563.2463379, 798.6261597, -1206.6101074, 1138.6119385
3: -498.7056580, 670.1018677, -696.4406738, 929.5416260, -1428.2473145, 1366.5419922
4: -440.1969910, 657.3359375, -606.9240723, 913.6280518, -1353.8250732, 1264.2600098

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6603363, upper bound: 886.6716574
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6603363, upper bound: 886.6716574
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -476.7803650, 771.6308594, -334.1739502, 545.4487305, -1022.2289429, 1105.8046875
1: -539.9763184, 772.4961548, -375.3325806, 539.6484375, -1079.6247559, 1147.8287354
2: -536.3087158, 761.8306885, -375.9838562, 533.0645142, -1069.3732910, 1137.8142090
3: -664.6658936, 886.8717041, -460.8188782, 621.6968994, -1286.3627930, 1347.6905518
4: -577.7337036, 871.2992554, -406.2091675, 607.7661743, -1185.4997559, 1277.5081787

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6344771, upper bound: 886.6295829
time: 0.93 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6373408, upper bound: 886.6430472
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -480.2467346, 776.7702637, -334.1739502, 545.4487305, -1025.6953125, 1110.9440918
1: -543.8629150, 775.5766602, -375.3325806, 539.6484375, -1083.5113525, 1150.9091797
2: -540.5451050, 765.2640991, -375.9838562, 533.0645142, -1073.6094971, 1141.2476807
3: -668.1536255, 890.8625488, -460.8188782, 621.6968994, -1289.8505859, 1351.6813965
4: -583.1776733, 875.6425781, -406.2091675, 607.7661743, -1190.9438477, 1281.8516846

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6344771, upper bound: 886.6347583
time: 0.98 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6373408, upper bound: 886.6482162
time: 0.95 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -476.7803650, 771.6308594, -358.9269409, 583.0151367, -1059.7954102, 1130.5576172
1: -539.9763184, 772.4961548, -402.1892090, 575.7573853, -1115.7336426, 1174.6853027
2: -536.3087158, 761.8306885, -403.6869507, 569.3664551, -1105.6751709, 1165.5174561
3: -664.6658936, 886.8717041, -493.3361206, 663.3167114, -1327.9826660, 1380.2073975
4: -577.7337036, 871.2992554, -435.9633789, 650.3424072, -1228.0758057, 1307.2620850

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6456963, upper bound: 886.6661974
time: 1.06 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6428388, upper bound: 886.6600579
time: 1.03 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6414292, upper bound: 886.6708779
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -480.2467346, 776.7702637, -358.9269409, 583.0151367, -1063.2618408, 1135.6968994
1: -543.8629150, 775.5766602, -402.1892090, 575.7573853, -1119.6203613, 1177.7657471
2: -540.5451050, 765.2640991, -403.6869507, 569.3664551, -1109.9113770, 1168.9508057
3: -668.1536255, 890.8625488, -493.3361206, 663.3167114, -1331.4703369, 1384.1984863
4: -583.1776733, 875.6425781, -435.9633789, 650.3424072, -1233.5200195, 1311.6055908

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6456963, upper bound: 886.6737980
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6428388, upper bound: 886.6649173
time: 1.14 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6414292, upper bound: 886.6707974
time: 1.11 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -483.8485718, 783.6398926, -342.6009827, 558.7576904, -1042.6062012, 1126.2408447
1: -547.6936646, 784.4992065, -384.6203613, 552.9998169, -1100.6933594, 1169.1196289
2: -544.0818481, 773.7381592, -385.3064575, 546.3264160, -1090.4082031, 1159.0441895
3: -674.3672485, 900.7269897, -472.4100342, 637.0861816, -1311.4533691, 1373.1369629
4: -585.6768188, 884.7250366, -415.8305969, 623.1256104, -1208.8023682, 1300.5555420

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6610293, upper bound: 886.6622535
time: 0.91 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6604523, upper bound: 886.6625027
time: 1.05 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -490.4929810, 794.2290649, -342.6009827, 558.7576904, -1049.2507324, 1136.8300781
1: -555.0930786, 793.1872559, -384.6203613, 552.9998169, -1108.0927734, 1177.8076172
2: -552.0131836, 782.6972656, -385.3064575, 546.3264160, -1098.3395996, 1168.0035400
3: -682.3418579, 911.2104492, -472.4100342, 637.0861816, -1319.4279785, 1383.6204834
4: -595.0556030, 895.1871338, -415.8305969, 623.1256104, -1218.1810303, 1311.0177002

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6610293, upper bound: 886.6629655
time: 0.95 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6604523, upper bound: 886.6632244
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -483.8485718, 783.6398926, -368.7682800, 598.6347046, -1082.4832764, 1152.4082031
1: -547.6936646, 784.4992065, -413.0889893, 591.4052734, -1139.0988770, 1197.5881348
2: -544.0818481, 773.7381592, -414.6584473, 584.9446411, -1129.0263672, 1188.3964844
3: -674.3672485, 900.7269897, -506.9190063, 681.3372192, -1355.7043457, 1407.6459961
4: -585.6768188, 884.7250366, -447.3838501, 668.3334961, -1254.0102539, 1332.1087646

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6610293, upper bound: 886.6622535
time: 0.98 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6604523, upper bound: 886.6690237
time: 1.00 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -490.4929810, 794.2290649, -368.7682800, 598.6347046, -1089.1276855, 1162.9973145
1: -555.0930786, 793.1872559, -413.0889893, 591.4052734, -1146.4982910, 1206.2762451
2: -552.0131836, 782.6972656, -414.6584473, 584.9446411, -1136.9575195, 1197.3557129
3: -682.3418579, 911.2104492, -506.9190063, 681.3372192, -1363.6790771, 1418.1293945
4: -595.0556030, 895.1871338, -447.3838501, 668.3334961, -1263.3891602, 1342.5710449

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6513909, upper bound: 886.6636677
time: 1.02 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6499599, upper bound: 886.6634526
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -476.7803650, 771.6308594, -487.9498291, 789.5354614, -1266.0357666, 1259.3250732
1: -539.9763184, 772.4961548, -552.3477173, 790.6480103, -1328.3171387, 1322.5137939
2: -536.3087158, 761.8306885, -548.6082764, 779.7769775, -1314.2335205, 1308.5858154
3: -664.6658936, 886.8717041, -680.1488647, 907.6591797, -1570.0465088, 1564.7233887
4: -577.7337036, 871.2992554, -590.4536743, 891.9060669, -1466.7751465, 1458.8840332

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6342474, upper bound: 886.6293071
time: 0.96 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6367832, upper bound: 886.6416089
time: 1.16 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -480.2467346, 776.7702637, -487.9498291, 789.5354614, -1269.7705078, 1264.7198486
1: -543.8629150, 775.5766602, -552.3477173, 790.6480103, -1332.4827881, 1326.3475342
2: -540.5451050, 765.2640991, -548.6082764, 779.7769775, -1318.5971680, 1312.8895264
3: -668.1536255, 890.8625488, -680.1488647, 907.6591797, -1573.8354492, 1569.6099854
4: -583.1776733, 875.6425781, -590.4536743, 891.9060669, -1472.4910889, 1463.8001709

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6342474, upper bound: 886.6345653
time: 0.98 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6367832, upper bound: 886.6468717
time: 1.13 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -476.7803650, 771.6308594, -492.9746704, 797.4037476, -1274.1839600, 1264.6054688
1: -539.9763184, 772.4961548, -557.9567261, 796.4893799, -1335.0064697, 1328.3924561
2: -536.3087158, 761.8306885, -554.6371460, 785.9524536, -1321.3038330, 1314.7943115
3: -664.6658936, 886.8717041, -685.8466187, 914.8530884, -1578.2264404, 1570.7573242
4: -577.7337036, 871.2992554, -597.7373047, 899.2791138, -1474.7249756, 1466.4660645

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6488741, upper bound: 886.6632870
time: 1.05 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6488741, upper bound: 886.6632870
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -480.2467346, 776.7702637, -492.9746704, 797.4037476, -1277.6505127, 1269.7448730
1: -543.8629150, 775.5766602, -557.9567261, 796.4893799, -1339.1719971, 1332.2260742
2: -540.5451050, 765.2640991, -554.6371460, 785.9524536, -1325.6677246, 1319.0979004
3: -668.1536255, 890.8625488, -685.8466187, 914.8530884, -1582.0153809, 1575.6439209
4: -583.1776733, 875.6425781, -597.7373047, 899.2791138, -1480.4719238, 1471.3975830

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6488741, upper bound: 886.6668744
time: 1.13 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6488741, upper bound: 886.6668744
time: 1.23 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -483.8485718, 783.6398926, -494.8212280, 800.8665771, -1284.4995117, 1278.2142334
1: -547.6936646, 784.4992065, -560.0187988, 802.0792236, -1347.6840820, 1342.3878174
2: -544.0818481, 773.7381592, -556.2733765, 791.0935059, -1333.4674072, 1328.2639160
3: -674.3672485, 900.7269897, -689.6235352, 920.7554932, -1593.2834473, 1588.4403076
4: -585.6768188, 884.7250366, -598.6124878, 904.7331543, -1487.4388428, 1480.4224854

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6418348, upper bound: 886.6321300
time: 0.95 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6443946, upper bound: 886.6443941
time: 0.99 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -490.4929810, 794.2290649, -494.8212280, 800.8665771, -1291.3593750, 1289.0502930
1: -555.0930786, 793.1872559, -560.0187988, 802.0792236, -1355.3352051, 1351.8662109
2: -552.0131836, 782.6972656, -556.2733765, 791.0935059, -1341.5466309, 1338.0634766
3: -682.3418579, 911.2104492, -689.6235352, 920.7554932, -1601.5578613, 1599.8815918
4: -595.0556030, 895.1871338, -598.6124878, 904.7331543, -1497.0299072, 1491.4173584

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6342474, upper bound: 886.6341232
time: 0.95 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6367832, upper bound: 886.6464600
time: 1.29 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -483.8485718, 783.6398926, -500.6353149, 810.0427246, -1293.8911133, 1284.2751465
1: -547.6936646, 784.4992065, -566.5096436, 809.3057861, -1355.7141113, 1349.1363525
2: -544.0818481, 773.7381592, -563.2463379, 798.6261597, -1341.8585205, 1335.3991699
3: -674.3672485, 900.7269897, -696.4406738, 929.5416260, -1603.0416260, 1595.5825195
4: -585.6768188, 884.7250366, -606.9240723, 913.6280518, -1496.8898926, 1488.9552002

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6505726, upper bound: 886.6662007
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6505726, upper bound: 886.6662007
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -490.4929810, 794.2290649, -500.6353149, 810.0427246, -1300.5355225, 1294.8643799
1: -555.0930786, 793.1872559, -566.5096436, 809.3057861, -1363.3651123, 1358.6147461
2: -552.0131836, 782.6972656, -563.2463379, 798.6261597, -1349.9377441, 1345.1986084
3: -682.3418579, 911.2104492, -696.4406738, 929.5416260, -1611.3161621, 1607.0239258
4: -595.0556030, 895.1871338, -606.9240723, 913.6280518, -1506.4866943, 1499.9549561

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6505726, upper bound: 886.6666825
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6488741, upper bound: 886.6666825
time: 0.99 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 5.48 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6764788, upper bound: 886.6791062
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6767064, upper bound: 886.6759886
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6764788, upper bound: 886.6791062
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6767064, upper bound: 886.6759886
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6841234, upper bound: 886.6822169
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6841234, upper bound: 886.6822169
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6841234, upper bound: 886.6822169
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6841234, upper bound: 886.6822169
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6646718, upper bound: 886.6686843
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6831418, upper bound: 886.6840579
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6646718, upper bound: 886.6686843
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6831418, upper bound: 886.6840579
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6610907, upper bound: 886.6378088
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6831418, upper bound: 886.6831418
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6610907, upper bound: 886.6378088
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6831418, upper bound: 886.6831418
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6596454, upper bound: 886.6374741
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6648529, upper bound: 886.6503966
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6586119, upper bound: 886.6361233
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6620789, upper bound: 886.6484595
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6757273, upper bound: 886.6695775
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6757273, upper bound: 886.6695775
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6706530, upper bound: 886.6670779
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6706530, upper bound: 886.6670779
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6631920, upper bound: 886.6379143
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6672226, upper bound: 886.6506555
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6575481, upper bound: 886.6360001
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6600135, upper bound: 886.6483535
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6603363, upper bound: 886.6680516
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6603363, upper bound: 886.6680516
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6603363, upper bound: 886.6716574
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6603363, upper bound: 886.6716574
IS_A2_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6344771, upper bound: 886.6295829
IS_A2_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6373408, upper bound: 886.6430472
IS_A2_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6344771, upper bound: 886.6347583
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6373408, upper bound: 886.6482162
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6428388, upper bound: 886.6600579
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6414292, upper bound: 886.6708779
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6428388, upper bound: 886.6649173
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6414292, upper bound: 886.6707974
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6610293, upper bound: 886.6622535
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6604523, upper bound: 886.6625027
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6610293, upper bound: 886.6629655
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6604523, upper bound: 886.6632244
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6610293, upper bound: 886.6622535
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6604523, upper bound: 886.6690237
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6513909, upper bound: 886.6636677
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6499599, upper bound: 886.6634526
IS_A2_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6342474, upper bound: 886.6293071
IS_A2_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6367832, upper bound: 886.6416089
IS_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6342474, upper bound: 886.6345653
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6367832, upper bound: 886.6468717
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6488741, upper bound: 886.6632870
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6488741, upper bound: 886.6632870
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6488741, upper bound: 886.6668744
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6488741, upper bound: 886.6668744
IS_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6418348, upper bound: 886.6321300
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6443946, upper bound: 886.6443941
IS_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6342474, upper bound: 886.6341232
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6367832, upper bound: 886.6464600
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6505726, upper bound: 886.6662007
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6505726, upper bound: 886.6662007
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6505726, upper bound: 886.6666825
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.48
Output dim: 0, lower bound: -886.6488741, upper bound: 886.6666825

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -306.4266052, 492.1778259, -308.1498413, 493.1778870, -799.6044312, 800.3275757
1: -343.2139893, 486.7157288, -344.5060120, 488.6116638, -831.8256226, 831.2217407
2: -345.0415039, 481.8312378, -346.5334778, 483.6400452, -828.6814575, 828.3646851
3: -420.3021240, 560.4561157, -421.9389038, 562.8870239, -983.1891479, 982.3950195
4: -373.0722046, 551.5798340, -374.3677979, 553.8787842, -926.9509888, 925.9476318

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6614221, upper bound: 886.6510668
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6690033, upper bound: 886.6661596
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6754149, upper bound: 886.6791062
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6763735, upper bound: 886.6789750
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -302.6304932, 486.1849670, -350.4417419, 574.3447266, -876.9752197, 836.6267090
1: -339.4423828, 480.9190979, -391.8486938, 565.0050049, -904.4473877, 872.7678223
2: -340.8238220, 476.0304565, -393.7427979, 559.0010986, -899.8249512, 869.7731934
3: -415.7210999, 553.5029297, -480.9304504, 651.9063110, -1067.6274414, 1034.4333496
4: -368.9042053, 545.0237427, -423.5451355, 636.6209106, -1005.5251465, 968.5688477

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6754241, upper bound: 886.6759886
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6763827, upper bound: 886.6758665
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -488.6704712, 792.5964355, -308.1498413, 493.1778870, -981.8483276, 1100.7462158
1: -550.5938110, 791.4447632, -344.5060120, 488.6116638, -1039.2053223, 1135.9508057
2: -549.6118774, 782.0778198, -346.5334778, 483.6400452, -1033.2517090, 1128.6113281
3: -677.1720581, 907.2698364, -421.9389038, 562.8870239, -1240.0590820, 1329.2084961
4: -591.4095459, 895.0852661, -374.3677979, 553.8787842, -1145.2883301, 1268.5972900

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6688087, upper bound: 886.6661596
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6761991, upper bound: 886.6777615
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6761991, upper bound: 886.6785873
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -481.6374207, 781.9213257, -350.4417419, 574.3447266, -1054.9700928, 1132.3629150
1: -542.7700195, 780.4518433, -391.8486938, 565.0050049, -1107.4825439, 1172.3002930
2: -541.6947021, 771.2475586, -393.7427979, 559.0010986, -1100.6958008, 1164.9902344
3: -667.2274780, 894.7072754, -480.9304504, 651.9063110, -1317.2044678, 1375.6375732
4: -583.5325317, 882.7197266, -423.5451355, 636.6209106, -1220.1534424, 1305.7109375

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6764785, upper bound: 886.6745542
time: 1.09 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6764785, upper bound: 886.6753817
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -309.7154236, 497.5668945, -491.5893860, 797.4088135, -1107.1242676, 989.1561279
1: -346.9890747, 492.0843201, -553.8572998, 796.1446533, -1143.1335449, 1045.9414062
2: -348.7414856, 487.0979004, -552.8887329, 786.7453003, -1135.4868164, 1039.9865723
3: -425.0345459, 566.6265869, -681.1791992, 912.6603394, -1337.6945801, 1247.8057861
4: -376.9709473, 557.5439453, -594.9200439, 900.4223022, -1276.9277344, 1152.4637451

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6785533, upper bound: 886.6756178
time: 1.24 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6765208, upper bound: 886.6757710
time: 1.05 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -309.7154236, 497.5668945, -492.3429565, 798.1012573, -1107.8166504, 989.9097290
1: -346.9890747, 492.0843201, -554.5442505, 796.8940430, -1143.8830566, 1046.6282959
2: -348.7414856, 487.0979004, -553.4963379, 787.6232910, -1136.3646240, 1040.5942383
3: -425.0345459, 566.6265869, -682.4190674, 913.6898193, -1338.7243652, 1249.0456543
4: -376.9709473, 557.5439453, -595.0189209, 901.7127686, -1278.1533203, 1152.5626221

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6785533, upper bound: 886.6756696
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6765208, upper bound: 886.6758226
time: 1.06 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -491.5893860, 797.4088135, -491.5893860, 797.4088135, -1288.9981689, 1288.9981689
1: -553.8572998, 796.1446533, -553.8572998, 796.1446533, -1348.4188232, 1348.4189453
2: -552.8887329, 786.7453003, -552.8887329, 786.7453003, -1338.0324707, 1338.0324707
3: -681.1791992, 912.6603394, -681.1791992, 912.6603394, -1592.8992920, 1592.8994141
4: -594.9200439, 900.4223022, -594.9200439, 900.4223022, -1492.8859863, 1492.8861084

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6836014, upper bound: 886.6808574
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6835638, upper bound: 886.6816473
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -491.5893860, 797.4088135, -492.3429565, 798.1012573, -1289.6906738, 1289.7515869
1: -553.8572998, 796.1446533, -554.5442505, 796.8940430, -1349.1229248, 1349.1176758
2: -552.8887329, 786.7453003, -553.4963379, 787.6232910, -1338.8892822, 1338.6680908
3: -681.1791992, 912.6603394, -682.4190674, 913.6898193, -1593.9230957, 1594.1632080
4: -594.9200439, 900.4223022, -595.0189209, 901.7127686, -1494.1115723, 1492.8774414

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6836014, upper bound: 886.6808574
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6835638, upper bound: 886.6816473
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -316.2154846, 506.8120728, -298.5121155, 479.3663940, -795.5818481, 805.3242188
1: -353.5516357, 502.1632080, -334.9028625, 475.9711609, -829.5228271, 837.0660400
2: -355.2905273, 497.3463440, -335.3381653, 470.3818970, -825.6724243, 832.6845093
3: -433.8125610, 578.5858765, -411.0116882, 548.1768188, -981.9893799, 989.5975342
4: -382.5516052, 568.5064087, -361.2874756, 538.0591431, -920.6107178, 929.7938843

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6759487, upper bound: 886.6685413
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6698798, upper bound: 886.6653704
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -324.3305359, 519.3591309, -328.3293457, 525.9785767, -850.3090820, 847.6884766
1: -362.7073364, 514.5890503, -367.2825928, 521.2624512, -883.9697266, 881.8716431
2: -364.5272827, 509.7775879, -369.0022888, 516.1498413, -880.6770630, 878.7799072
3: -444.8699646, 592.8459473, -450.4012146, 600.4615479, -1045.3315430, 1043.2471924
4: -392.8623047, 583.0898438, -397.7448120, 590.4666748, -983.3289795, 980.8346558

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6687083, upper bound: 886.6792065
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6687083, upper bound: 886.6840579
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -485.9708557, 788.5503540, -298.5121155, 479.3663940, -965.3372192, 1087.0625000
1: -547.4649048, 787.4879150, -334.9028625, 475.9711609, -1023.4360352, 1122.3907471
2: -546.2861938, 778.2014160, -335.3381653, 470.3818970, -1016.6680298, 1113.5395508
3: -674.0090942, 902.8925781, -411.0116882, 548.1768188, -1222.1859131, 1313.9042969
4: -587.1948242, 890.8352051, -361.2874756, 538.0591431, -1125.2539062, 1251.3986816

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6619266, upper bound: 886.6685413
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6339663, upper bound: 886.6508912
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6339663, upper bound: 886.6686843
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -491.7754822, 797.1674805, -328.3293457, 525.9785767, -1017.7540283, 1125.4965820
1: -553.9066162, 795.9835815, -367.2825928, 521.2624512, -1075.1689453, 1163.2661133
2: -552.8540039, 786.7182617, -369.0022888, 516.1498413, -1069.0039062, 1155.7205811
3: -681.6381226, 912.6412354, -450.4012146, 600.4615479, -1282.0996094, 1363.0424805
4: -594.3188477, 900.6728516, -397.7448120, 590.4666748, -1184.7855225, 1297.7706299

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6778642, upper bound: 886.6749511
time: 1.10 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6378088, upper bound: 886.6610907
time: 1.15 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6378088, upper bound: 886.6840579
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -316.2154846, 506.8120728, -481.0843506, 782.8259277, -1099.0412598, 987.8964233
1: -353.5516357, 502.1632080, -542.6111450, 782.3541260, -1135.9057617, 1044.7742920
2: -355.2905273, 497.3463440, -540.8375854, 772.8110352, -1128.1015625, 1038.1837158
3: -433.8125610, 578.5858765, -668.1291504, 896.7509155, -1330.5633545, 1246.7148438
4: -382.5516052, 568.5064087, -581.8550415, 884.3529053, -1265.8913574, 1150.3613281

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6492459, upper bound: 886.6145261
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6519207, upper bound: 886.6154111
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -324.3305359, 519.3591309, -502.3356934, 814.6252441, -1138.9558105, 1021.6948242
1: -362.7073364, 514.5890503, -565.8380737, 813.7943115, -1176.5017090, 1080.4271240
2: -364.5272827, 509.7775879, -564.6455688, 804.2678833, -1168.7951660, 1074.4230957
3: -444.8699646, 592.8459473, -696.3634033, 933.0020752, -1377.8720703, 1289.2093506
4: -392.8623047, 583.0898438, -606.9974365, 920.5051270, -1312.4869385, 1190.0871582

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6686843, upper bound: 886.6646718
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6686843, upper bound: 886.6831418
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -485.9708557, 788.5503540, -481.0843506, 782.8259277, -1268.7965088, 1269.6347656
1: -547.4649048, 787.4879150, -542.6111450, 782.3541260, -1328.3289795, 1328.6202393
2: -546.2861938, 778.2014160, -540.8375854, 772.8110352, -1317.4979248, 1317.5339355
3: -674.0090942, 902.8925781, -668.1291504, 896.7509155, -1570.2830811, 1570.2641602
4: -587.1948242, 890.8352051, -581.8550415, 884.3529053, -1468.5909424, 1469.9262695

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6320111, upper bound: 886.6320112
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6320111, upper bound: 886.6378088
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -491.7754822, 797.1674805, -502.3356934, 814.6252441, -1306.4007568, 1299.5031738
1: -553.9066162, 795.9835815, -565.8380737, 813.7943115, -1366.2410889, 1360.4040527
2: -552.8540039, 786.7182617, -564.6455688, 804.2678833, -1355.4627686, 1349.9097900
3: -681.6381226, 912.6412354, -696.3634033, 933.0020752, -1614.0043945, 1608.2238770
4: -594.3188477, 900.6728516, -606.9974365, 920.5051270, -1512.1025391, 1505.1932373

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6378088, upper bound: 886.6610907
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6378088, upper bound: 886.6831418
time: 1.33 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -316.1903076, 515.8814697, -450.1239929, 727.4001465, -1043.5903320, 966.0053711
1: -354.5682373, 508.7829590, -508.6058655, 728.9163208, -1083.4843750, 1017.2415771
2: -355.9424438, 503.0243835, -505.2940979, 718.6322632, -1074.4367676, 1008.3184814
3: -435.0342407, 586.0864258, -628.2235107, 838.4852295, -1273.5192871, 1214.3099365
4: -384.6293640, 573.5560913, -542.2988281, 822.0863647, -1205.3476562, 1115.2191162

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6516037, upper bound: 886.6302965
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6513453, upper bound: 886.6296488
time: 0.94 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -332.6294556, 542.6247559, -485.4176025, 785.4420166, -1118.0715332, 1028.0421143
1: -373.2080383, 535.1131592, -549.5399780, 786.5379028, -1159.7459717, 1084.5507812
2: -374.5007629, 529.1209717, -545.7509155, 775.7050781, -1150.2058105, 1074.8718262
3: -457.5836487, 616.3934326, -676.6870117, 902.9337769, -1360.5174561, 1293.0804443
4: -405.0903931, 603.3582153, -587.3999634, 887.2642212, -1291.4154053, 1190.6142578

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6566646, upper bound: 886.6421124
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6534616, upper bound: 886.6411242
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -333.1303711, 547.3335571, -440.1072388, 712.0161743, -1045.1464844, 987.4407959
1: -372.7014465, 538.4318848, -496.9454041, 712.6862793, -1085.3874512, 1035.3773193
2: -375.0294495, 532.6110840, -493.9635620, 702.4730225, -1077.5021973, 1026.5747070
3: -457.3345032, 620.5858154, -613.7438354, 820.4059448, -1277.7403564, 1234.3295898
4: -403.1492615, 606.3898315, -529.6795654, 803.1184692, -1206.2677002, 1136.0693359

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6509968, upper bound: 886.6288183
time: 1.07 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6497179, upper bound: 886.6285583
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -349.0325012, 573.4271240, -474.8290405, 769.2191772, -1118.2517090, 1048.2561035
1: -390.7023010, 563.7913818, -537.2882690, 769.4774170, -1160.1795654, 1101.0795898
2: -392.9434814, 557.7009277, -533.7620239, 758.7352905, -1151.6785889, 1091.4628906
3: -479.0653992, 649.7391968, -661.5482178, 883.8156738, -1362.8811035, 1311.2871094
4: -422.7808533, 634.7658081, -574.0272217, 867.4686890, -1290.2495117, 1208.7928467

Time for backsubstitution: 1.89 seconds
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=1024.72607421875
rel_dist={0: [-886.6909129349328, 886.6909129349328]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6778859, upper bound: 886.6764083
time: 1.24 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6737988, upper bound: 886.6737988
time: 1.02 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.41 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.41
Output dim: 0, lower bound: -886.6778859, upper bound: 886.6764083
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.41
Output dim: 0, lower bound: -886.6737988, upper bound: 886.6737988

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -373.4723206, 606.2720947, -381.1879578, 618.9174805, -992.3897705, 987.4600220
1: -418.3924866, 599.0228271, -427.0381470, 611.6237183, -1030.0162354, 1026.0610352
2: -419.9433899, 592.4771118, -428.6359253, 605.0542603, -1024.9973145, 1021.1130371
3: -513.4045410, 690.0853882, -524.0444946, 704.4754639, -1217.8800049, 1214.1297607
4: -453.0448608, 677.0018921, -462.1968079, 691.2159424, -1144.2607422, 1139.1987305

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6737988, upper bound: 886.6737988
time: 0.76 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6737988, upper bound: 886.6737988
time: 1.04 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -506.5198975, 819.3720703, -367.9176636, 596.4960327, -1103.0158691, 1187.2897949
1: -573.1215820, 818.7601929, -412.0688782, 590.4548340, -1163.5764160, 1230.8289795
2: -569.8142090, 807.9828491, -413.5782776, 584.3431396, -1154.1571045, 1221.5611572
3: -704.5859375, 940.3079834, -505.8311157, 680.5980835, -1385.1839600, 1446.1389160
4: -613.9079590, 924.3903198, -445.2311096, 667.0209351, -1280.9288330, 1369.6214600

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6737988, upper bound: 886.6737988
time: 1.21 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6737988, upper bound: 886.6737988
time: 0.87 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.16 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.16
Output dim: 0, lower bound: -886.6737988, upper bound: 886.6737988
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.16
Output dim: 0, lower bound: -886.6737988, upper bound: 886.6737988
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.16
Output dim: 0, lower bound: -886.6737988, upper bound: 886.6737988
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.16
Output dim: 0, lower bound: -886.6737988, upper bound: 886.6737988

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -373.4723206, 606.2720947, -373.4723206, 606.2720947, -979.7443848, 979.7443848
1: -418.3924866, 599.0228271, -418.3924866, 599.0228271, -1017.4151001, 1017.4151611
2: -419.9433899, 592.4771118, -419.9433899, 592.4771118, -1012.4205322, 1012.4204712
3: -513.4045410, 690.0853882, -513.4045410, 690.0853882, -1203.4899902, 1203.4899902
4: -453.0448608, 677.0018921, -453.0448608, 677.0018921, -1130.0467529, 1130.0467529

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6715825, upper bound: 886.6680195
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6724290, upper bound: 886.6680195
time: 0.87 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -373.4723206, 606.2720947, -506.5198975, 819.3720703, -1192.8441162, 1112.7918701
1: -418.3924866, 599.0228271, -573.1215820, 818.7601929, -1237.1527100, 1172.1444092
2: -419.9433899, 592.4771118, -569.8142090, 807.9828491, -1227.9262695, 1162.2911377
3: -513.4045410, 690.0853882, -704.5859375, 940.3079834, -1453.7124023, 1394.6711426
4: -453.0448608, 677.0018921, -613.9079590, 924.3903198, -1377.4351807, 1290.9099121

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6715825, upper bound: 886.6680195
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6724290, upper bound: 886.6680195
time: 0.79 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -506.5198975, 819.3720703, -373.4721375, 606.2718506, -1112.7916260, 1192.8439941
1: -573.1215820, 818.7601929, -418.3923035, 599.0225220, -1172.1440430, 1237.1524658
2: -569.8142090, 807.9828491, -419.9431458, 592.4768066, -1162.2910156, 1227.9260254
3: -704.5859375, 940.3079834, -513.4042358, 690.0850220, -1394.6708984, 1453.7121582
4: -613.9079590, 924.3903198, -453.0445862, 677.0014648, -1290.9094238, 1377.4349365

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6694860, upper bound: 886.6649594
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6649594, upper bound: 886.6649594
time: 1.02 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -506.5198975, 819.3720703, -506.5198975, 819.3720703, -1325.8917236, 1325.8918457
1: -573.1215820, 818.7601929, -573.1215820, 818.7601929, -1391.2940674, 1391.2940674
2: -569.8142090, 807.9828491, -569.8142090, 807.9828491, -1377.6153564, 1377.6153564
3: -704.5859375, 940.3079834, -704.5859375, 940.3079834, -1644.8936768, 1644.8936768
4: -613.9079590, 924.3903198, -613.9079590, 924.3903198, -1536.3483887, 1536.3483887

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6694860, upper bound: 886.6649594
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6649594, upper bound: 886.6649594
time: 1.01 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.90 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.90
Output dim: 0, lower bound: -886.6715825, upper bound: 886.6680195
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.90
Output dim: 0, lower bound: -886.6724290, upper bound: 886.6680195
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.90
Output dim: 0, lower bound: -886.6715825, upper bound: 886.6680195
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.90
Output dim: 0, lower bound: -886.6724290, upper bound: 886.6680195
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.90
Output dim: 0, lower bound: -886.6694860, upper bound: 886.6649594
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.90
Output dim: 0, lower bound: -886.6649594, upper bound: 886.6649594
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.90
Output dim: 0, lower bound: -886.6694860, upper bound: 886.6649594
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.90
Output dim: 0, lower bound: -886.6649594, upper bound: 886.6649594

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -349.6350098, 569.3693848, -352.3813171, 572.5598145, -922.1948242, 921.7507324
1: -392.1717224, 561.8006592, -394.8960266, 565.1838989, -957.3554077, 956.6966553
2: -393.7890320, 555.6325073, -396.4107056, 558.8980713, -952.6870728, 952.0430908
3: -480.6766052, 646.9263306, -484.2410278, 651.1449585, -1131.8212891, 1131.1671143
4: -425.8119507, 634.2166138, -428.3787842, 638.2525635, -1064.0644531, 1062.5954590

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6829533, upper bound: 886.6811965
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6829321, upper bound: 886.6802686
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -364.6583252, 591.8895874, -367.8742065, 597.1077881, -961.7661133, 959.7637329
1: -408.4007263, 584.5307007, -412.0494690, 589.7830811, -998.1838379, 996.5802002
2: -410.0544434, 578.2492065, -413.6636353, 583.3924561, -993.4468384, 991.9128418
3: -501.2101135, 673.4092407, -505.6437378, 679.4520264, -1180.6621094, 1179.0529785
4: -442.3909302, 660.6585083, -446.2645569, 666.5569458, -1108.9477539, 1106.9230957

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6813299, upper bound: 886.6820305
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6813090, upper bound: 886.6813090
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -349.6350098, 569.3693848, -487.9803772, 789.2459717, -1138.8809814, 1057.3497314
1: -392.1717224, 561.8006592, -552.3669434, 788.1997070, -1180.3713379, 1114.1673584
2: -393.7890320, 555.6325073, -549.0773926, 777.7661743, -1171.5551758, 1104.7095947
3: -480.6766052, 646.9263306, -678.8982544, 905.3513184, -1386.0274658, 1325.8244629
4: -425.8119507, 634.2166138, -591.8535156, 889.9547119, -1315.7664795, 1226.0700684

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6681153, upper bound: 886.6611276
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6713030, upper bound: 886.6680195
time: 1.21 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -364.6583252, 591.8895874, -498.9787292, 807.5554199, -1172.2137451, 1090.8682861
1: -408.4007263, 584.5307007, -564.6331787, 806.7137451, -1215.1145020, 1149.1638184
2: -410.0544434, 578.2492065, -561.4624634, 796.0758667, -1206.1301270, 1139.7116699
3: -501.2101135, 673.4092407, -694.1079712, 926.6065674, -1427.8166504, 1367.5172119
4: -442.3909302, 660.6585083, -605.0621338, 910.6511230, -1353.0419922, 1265.7207031

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6712821, upper bound: 886.6629846
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6719840, upper bound: 886.6680195
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -482.6397095, 780.5917358, -352.3811340, 572.5596313, -1055.1990967, 1132.9729004
1: -546.5246582, 779.3952637, -394.8958740, 565.1835938, -1111.7081299, 1174.2911377
2: -543.2442627, 769.0665283, -396.4104919, 558.8978882, -1102.1419678, 1165.4769287
3: -671.4299927, 895.2281494, -484.2408447, 651.1447144, -1322.5743408, 1379.4689941
4: -586.0380249, 880.0111694, -428.3784790, 638.2523193, -1224.2901611, 1308.3896484

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6675204, upper bound: 886.6605688
time: 1.08 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6706929, upper bound: 886.6719738
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -492.6635437, 797.7097778, -367.8740540, 597.1074829, -1089.7709961, 1165.5837402
1: -557.5224609, 796.6813965, -412.0492249, 589.7828369, -1147.3052979, 1208.7305908
2: -554.4656372, 786.1628418, -413.6634521, 583.3920898, -1137.8576660, 1199.8262939
3: -685.3289185, 915.2006226, -505.6434937, 679.4517212, -1364.7806396, 1420.8441162
4: -597.6687012, 899.1670532, -446.2643433, 666.5566406, -1264.2252197, 1345.4313965

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6648656, upper bound: 886.6614360
time: 1.00 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6680195, upper bound: 886.6719840
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -482.6397095, 780.5917358, -487.9803772, 789.2459717, -1271.8856201, 1268.5721436
1: -546.5246582, 779.3952637, -552.3669434, 788.1997070, -1334.0170898, 1330.8437500
2: -543.2442627, 769.0665283, -549.0773926, 777.7661743, -1320.6760254, 1317.7960205
3: -671.4299927, 895.2281494, -678.8982544, 905.3513184, -1576.5157471, 1573.6556396
4: -586.0380249, 880.0111694, -591.8535156, 889.9547119, -1474.2563477, 1470.1358643

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6675000, upper bound: 886.6594845
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6691517, upper bound: 886.6647744
time: 1.06 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -492.6635437, 797.7097778, -498.9787292, 807.5554199, -1300.2189941, 1296.6884766
1: -557.5224609, 796.6813965, -564.6331787, 806.7137451, -1363.6314697, 1360.6820068
2: -554.4656372, 786.1628418, -561.4624634, 796.0758667, -1350.2868652, 1347.3419189
3: -685.3289185, 915.2006226, -694.1079712, 926.6065674, -1611.9355469, 1609.3085938
4: -597.6687012, 899.1670532, -605.0621338, 910.6511230, -1506.2998047, 1502.2236328

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6642197, upper bound: 886.6594811
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6647744, upper bound: 886.6647744
time: 1.26 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 5.20 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.20
Output dim: 0, lower bound: -886.6829533, upper bound: 886.6811965
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.20
Output dim: 0, lower bound: -886.6829321, upper bound: 886.6802686
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.20
Output dim: 0, lower bound: -886.6813299, upper bound: 886.6820305
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.20
Output dim: 0, lower bound: -886.6813090, upper bound: 886.6813090
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.20
Output dim: 0, lower bound: -886.6681153, upper bound: 886.6611276
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.20
Output dim: 0, lower bound: -886.6713030, upper bound: 886.6680195
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.20
Output dim: 0, lower bound: -886.6712821, upper bound: 886.6629846
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.20
Output dim: 0, lower bound: -886.6719840, upper bound: 886.6680195
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.20
Output dim: 0, lower bound: -886.6675204, upper bound: 886.6605688
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.20
Output dim: 0, lower bound: -886.6706929, upper bound: 886.6719738
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.20
Output dim: 0, lower bound: -886.6648656, upper bound: 886.6614360
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.20
Output dim: 0, lower bound: -886.6680195, upper bound: 886.6719840
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.20
Output dim: 0, lower bound: -886.6675000, upper bound: 886.6594845
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.20
Output dim: 0, lower bound: -886.6691517, upper bound: 886.6647744
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.20
Output dim: 0, lower bound: -886.6642197, upper bound: 886.6594811
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.20
Output dim: 0, lower bound: -886.6647744, upper bound: 886.6647744

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -326.3413391, 526.4268188, -313.1058960, 501.6006470, -827.9419556, 839.5327148
1: -365.8748169, 520.3529053, -350.5122681, 496.8088684, -862.6837158, 870.8651123
2: -367.5196533, 515.0090332, -352.1573181, 491.7368774, -859.2565308, 867.1663818
3: -448.2516174, 599.1636963, -429.5734863, 571.7915649, -1020.0431519, 1028.7370605
4: -397.2829590, 589.1517334, -380.1655273, 562.9150391, -960.1979980, 969.3172607

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6829321, upper bound: 886.6802686
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6829321, upper bound: 886.6802686
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -341.8214111, 557.1486206, -496.3288269, 804.8297729, -1146.6511230, 1053.4774170
1: -383.7387695, 549.5902100, -559.1584473, 803.8197632, -1187.5584717, 1108.7485352
2: -385.0083923, 543.3743286, -558.0601807, 794.3797607, -1179.3879395, 1101.4345703
3: -470.1198120, 632.8974609, -687.8524780, 921.5256958, -1391.6452637, 1320.7500000
4: -416.8627930, 620.4782104, -600.1920776, 909.2183838, -1325.0797119, 1220.6702881

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6829321, upper bound: 886.6802686
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6829321, upper bound: 886.6802686
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -342.7408142, 551.5160522, -328.1974182, 525.6290894, -868.3698120, 879.7132568
1: -383.5744934, 545.9238281, -367.1173096, 520.8699341, -904.4444580, 913.0411377
2: -385.2955017, 540.6195068, -368.8658447, 515.9031372, -901.1986084, 909.4853516
3: -470.6098328, 628.8352051, -450.2304993, 600.0592651, -1070.6690674, 1079.0656738
4: -415.4502869, 618.2422485, -397.5299683, 590.1916504, -1005.6419067, 1015.7721558

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6813090, upper bound: 886.6813090
time: 1.04 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6813090, upper bound: 886.6813090
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -354.4755554, 575.4893799, -500.8932495, 812.2302856, -1166.7058105, 1076.3825684
1: -397.3623962, 568.1229248, -564.1883545, 811.1494751, -1208.5118408, 1132.3112793
2: -398.6248169, 561.8377075, -563.0802002, 801.6936646, -1200.3182373, 1124.9177246
3: -487.4856262, 654.4940796, -694.2907104, 930.0158081, -1417.5012207, 1348.7847900
4: -430.6034546, 642.3168945, -605.3632812, 917.6698608, -1347.4957275, 1247.6800537

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6813090, upper bound: 886.6813090
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6813090, upper bound: 886.6813090
time: 1.04 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -335.5285645, 546.8900146, -480.9851074, 778.2615967, -1113.7901611, 1027.8750000
1: -376.3257751, 539.9152222, -544.5513306, 779.2593994, -1155.5850830, 1084.4665527
2: -377.8156738, 533.7640991, -540.8483276, 768.5051880, -1146.3208008, 1074.6124268
3: -461.5285339, 621.7182007, -670.5003662, 894.6274414, -1356.1560059, 1292.2185059
4: -408.1346741, 609.0021362, -582.2208252, 879.0155640, -1287.1502686, 1191.2227783

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6599015, upper bound: 886.6537618
time: 1.10 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6604498, upper bound: 886.6541981
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -348.6113892, 567.7216797, -485.9121399, 785.9494629, -1134.5607910, 1053.6336670
1: -391.0217896, 560.1693726, -550.0541382, 784.9027710, -1175.9245605, 1110.2235107
2: -392.6341858, 554.0127563, -546.7421265, 774.4929199, -1167.1269531, 1100.7546387
3: -479.2770691, 645.0673218, -676.0552368, 901.5872192, -1380.8642578, 1321.1224365
4: -424.5957642, 632.3454590, -589.3737183, 886.1828003, -1310.7785645, 1221.7192383

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6693365, upper bound: 886.6662661
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6679562, upper bound: 886.6651721
time: 1.01 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -350.4916077, 569.1045532, -490.6319885, 794.2886353, -1144.7802734, 1059.7365723
1: -392.4512024, 562.3419189, -555.3120728, 795.3617554, -1187.8126221, 1117.6540527
2: -393.9978943, 556.3469238, -551.6184692, 784.4631958, -1178.4610596, 1107.9653320
3: -481.9555969, 647.9959717, -683.7963257, 913.0986328, -1395.0538330, 1331.7922363
4: -424.6148071, 635.2381592, -593.6692505, 897.0938721, -1321.7087402, 1228.9074707

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6697516, upper bound: 886.6624711
time: 5.37 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6667805, upper bound: 886.6607164
time: 1.14 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -363.8032532, 590.5147095, -496.7932739, 804.0512085, -1167.8544922, 1087.3077393
1: -407.4447937, 583.1825562, -562.1866455, 803.2003784, -1210.6451416, 1145.3691406
2: -409.0895996, 576.9039917, -558.9930420, 792.5903320, -1201.6799316, 1135.8969727
3: -500.0432434, 671.8663940, -691.1019897, 922.5942383, -1422.6370850, 1362.9683838
4: -441.3691101, 659.1085205, -602.4304199, 906.6475220, -1348.0166016, 1261.5389404

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6701913, upper bound: 886.6662661
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6676328, upper bound: 886.6652772
time: 1.38 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -468.9117432, 758.8192749, -326.4608459, 533.1658936, -1002.0776367, 1085.2800293
1: -531.2672729, 758.2286987, -366.7523804, 527.3087158, -1058.5758057, 1124.9809570
2: -527.7647095, 747.9273682, -367.4018860, 520.8673096, -1048.6319580, 1115.3292236
3: -652.9634399, 870.7859497, -450.2108459, 607.4971313, -1260.4603271, 1320.9967041
4: -569.2075806, 855.7059326, -397.2102661, 593.6644897, -1162.8720703, 1252.9162598

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6598308, upper bound: 886.6530557
time: 1.17 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6590318, upper bound: 886.6537698
time: 0.97 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -481.5160828, 778.7941895, -350.6332703, 569.7678833, -1051.2839355, 1129.4273682
1: -545.2738647, 777.5958862, -392.9486694, 562.4432373, -1107.7169189, 1170.5445557
2: -541.9755249, 767.2750854, -394.4375305, 556.1668701, -1098.1422119, 1161.7126465
3: -669.8903809, 893.1709595, -481.8733521, 648.0060425, -1317.8964844, 1375.0443115
4: -584.6905518, 877.9556274, -426.3060913, 635.1034546, -1219.7939453, 1304.2615967

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6537771, upper bound: 886.6659661
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6537771, upper bound: 886.6719738
time: 0.96 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -478.2946472, 774.8043823, -339.7637024, 554.1115112, -1032.4061279, 1114.5681152
1: -541.5545044, 774.3806152, -381.3984680, 548.3135986, -1089.8681641, 1155.7790527
2: -538.2458496, 763.8701782, -382.1158752, 541.7268677, -1079.9726562, 1145.9860840
3: -665.9706421, 889.4753418, -468.4788513, 631.6959229, -1297.6665039, 1357.9542236
4: -580.0510864, 873.6472168, -412.3956604, 617.8391724, -1197.8902588, 1286.0427246

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6604600, upper bound: 886.6614360
time: 0.93 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6604600, upper bound: 886.6614360
time: 1.02 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -491.6274719, 796.0514526, -366.0813599, 594.2271118, -1085.8546143, 1162.1326904
1: -556.3656616, 795.0169067, -410.0423889, 586.9569092, -1143.3225098, 1205.0592041
2: -553.2960205, 784.5112915, -411.6423035, 580.5726929, -1133.8686523, 1196.1535645
3: -683.9045410, 913.2982178, -503.1961975, 676.2191772, -1360.1235352, 1416.4943848
4: -596.4250488, 897.2702026, -444.1208801, 663.3105469, -1259.7355957, 1341.3911133

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6537771, upper bound: 886.6712821
time: 1.35 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6629846, upper bound: 886.6719840
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -468.9117432, 758.8192749, -480.9851074, 778.2615967, -1247.1732178, 1239.8043213
1: -531.2672729, 758.2286987, -544.5513306, 779.2593994, -1308.6606445, 1301.5974121
2: -527.7647095, 747.9273682, -540.8483276, 768.5051880, -1294.7579346, 1288.1926270
3: -652.9634399, 870.7859497, -670.5003662, 894.6274414, -1546.0041504, 1540.4127197
4: -569.2075806, 855.7059326, -582.2208252, 879.0155640, -1445.6940918, 1435.7541504

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6598134, upper bound: 886.6524734
time: 1.15 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6587849, upper bound: 886.6525468
time: 1.06 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -481.5160828, 778.7941895, -485.9121399, 785.9494629, -1267.4655762, 1264.7059326
1: -545.2738647, 777.5958862, -550.0541382, 784.9027710, -1329.1125488, 1326.4631348
2: -541.9755249, 767.2750854, -546.7421265, 774.4929199, -1315.7565918, 1313.3666992
3: -669.8903809, 893.1709595, -676.0552368, 901.5872192, -1570.6773682, 1568.2990723
4: -584.6905518, 877.9556274, -589.3737183, 886.1828003, -1468.9971924, 1465.4821777

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6516152, upper bound: 886.6610141
time: 1.30 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6516152, upper bound: 886.6647744
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -478.2946472, 774.8043823, -490.6319885, 794.2886353, -1272.5832520, 1265.4364014
1: -541.5545044, 774.3806152, -555.3120728, 795.3617554, -1335.2156982, 1328.8388672
2: -538.2458496, 763.8701782, -551.6184692, 784.4631958, -1321.3652344, 1315.0478516
3: -665.9706421, 889.4753418, -683.7963257, 913.0986328, -1577.9401855, 1573.0057373
4: -580.0510864, 873.6472168, -593.6692505, 897.0938721, -1474.4342041, 1465.0410156

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6509988, upper bound: 886.6589119
time: 0.96 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6589119, upper bound: 886.6594811
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -491.6274719, 796.0514526, -496.7932739, 804.0512085, -1295.6787109, 1292.8446045
1: -556.3656616, 795.0169067, -562.1866455, 803.2003784, -1358.5837402, 1356.2750244
2: -553.2960205, 784.5112915, -558.9930420, 792.5903320, -1345.2349854, 1342.9007568
3: -683.9045410, 913.2982178, -691.1019897, 922.5942383, -1606.0538330, 1603.9914551
4: -596.4250488, 897.2702026, -602.4304199, 906.6475220, -1500.8999023, 1497.5823975

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6594811, upper bound: 886.6642197
time: 0.99 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6594811, upper bound: 886.6647744
time: 1.21 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 5.68 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -886.6829321, upper bound: 886.6802686
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -886.6829321, upper bound: 886.6802686
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -886.6829321, upper bound: 886.6802686
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -886.6829321, upper bound: 886.6802686
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -886.6813090, upper bound: 886.6813090
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -886.6813090, upper bound: 886.6813090
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -886.6813090, upper bound: 886.6813090
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -886.6813090, upper bound: 886.6813090
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -886.6599015, upper bound: 886.6537618
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -886.6604498, upper bound: 886.6541981
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -886.6693365, upper bound: 886.6662661
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -886.6679562, upper bound: 886.6651721
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -886.6697516, upper bound: 886.6624711
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -886.6667805, upper bound: 886.6607164
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -886.6701913, upper bound: 886.6662661
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -886.6676328, upper bound: 886.6652772
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -886.6598308, upper bound: 886.6530557
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -886.6590318, upper bound: 886.6537698
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -886.6537771, upper bound: 886.6659661
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -886.6537771, upper bound: 886.6719738
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -886.6604600, upper bound: 886.6614360
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -886.6604600, upper bound: 886.6614360
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -886.6537771, upper bound: 886.6712821
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -886.6629846, upper bound: 886.6719840
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -886.6598134, upper bound: 886.6524734
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -886.6587849, upper bound: 886.6525468
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -886.6516152, upper bound: 886.6610141
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -886.6516152, upper bound: 886.6647744
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -886.6509988, upper bound: 886.6589119
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -886.6589119, upper bound: 886.6594811
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -886.6594811, upper bound: 886.6642197
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.68
Output dim: 0, lower bound: -886.6594811, upper bound: 886.6647744

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -309.7154236, 497.5668945, -313.1058960, 501.6006470, -811.3160400, 810.6727295
1: -346.9890747, 492.0843201, -350.5122681, 496.8088684, -843.7979736, 842.5964966
2: -348.7414856, 487.0979004, -352.1573181, 491.7368774, -840.4783325, 839.2552490
3: -425.0345459, 566.6265869, -429.5734863, 571.7915649, -996.8260498, 996.2000732
4: -376.9709473, 557.5439453, -380.1655273, 562.9150391, -939.8859863, 937.7093506

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6752950, upper bound: 886.6773695
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6755477, upper bound: 886.6742497
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -491.5893860, 797.4088135, -313.1058960, 501.6006470, -993.1899414, 1110.5146484
1: -553.8572998, 796.1446533, -350.5122681, 496.8088684, -1050.6661377, 1146.6568604
2: -552.8887329, 786.7453003, -352.1573181, 491.7368774, -1044.6254883, 1138.9025879
3: -681.1791992, 912.6603394, -429.5734863, 571.7915649, -1252.9707031, 1342.2333984
4: -594.9200439, 900.4223022, -380.1655273, 562.9150391, -1157.8350830, 1279.9815674

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6752950, upper bound: 886.6773695
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6752950, upper bound: 886.6742497
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -309.7154236, 497.5668945, -496.3288269, 804.8297729, -1114.5451660, 993.8955688
1: -346.9890747, 492.0843201, -559.1584473, 803.8197632, -1150.8087158, 1051.2425537
2: -348.7414856, 487.0979004, -558.0601807, 794.3797607, -1143.1212158, 1045.1580811
3: -425.0345459, 566.6265869, -687.8524780, 921.5256958, -1346.5601807, 1254.4790039
4: -376.9709473, 557.5439453, -600.1920776, 909.2183838, -1285.3957520, 1157.7357178

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6659212, upper bound: 886.6752549
time: 1.14 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6821580, upper bound: 886.6802686
time: 1.06 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6821580, upper bound: 886.6802686
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -491.5893860, 797.4088135, -496.3288269, 804.8297729, -1296.4191895, 1293.7376709
1: -553.8572998, 796.1446533, -559.1584473, 803.8197632, -1355.8157959, 1353.6256104
2: -552.8887329, 786.7453003, -558.0601807, 794.3797607, -1345.3598633, 1343.1359863
3: -681.1791992, 912.6603394, -687.8524780, 921.5256958, -1601.5350342, 1599.4317627
4: -594.9200439, 900.4223022, -600.1920776, 909.2183838, -1501.3537598, 1497.9660645

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6659212, upper bound: 886.6737498
time: 1.22 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6821580, upper bound: 886.6802686
time: 1.27 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6821580, upper bound: 886.6802686
time: 1.39 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -324.6807556, 519.9534912, -328.1974182, 525.6290894, -850.3098145, 848.1507568
1: -363.1039124, 515.1547241, -367.1173096, 520.8699341, -883.9737549, 882.2720337
2: -364.9215393, 510.3396912, -368.8658447, 515.9031372, -880.8247070, 879.2055054
3: -445.3571777, 593.4895020, -450.2304993, 600.0592651, -1045.4165039, 1043.7199707
4: -393.2788391, 583.7371826, -397.5299683, 590.1916504, -983.4704590, 981.2670898

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6498601, upper bound: 886.6556708
time: 1.68 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6813299, upper bound: 886.6820305
time: 1.44 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -492.3429565, 798.1012573, -328.1974182, 525.6290894, -1017.9720459, 1126.2987061
1: -554.5442505, 796.8940430, -367.1173096, 520.8699341, -1075.4141846, 1164.0113525
2: -553.4963379, 787.6232910, -368.8658447, 515.9031372, -1069.3994141, 1156.4887695
3: -682.4190674, 913.6898193, -450.2304993, 600.0592651, -1282.4782715, 1363.9202881
4: -595.0189209, 901.7127686, -397.5299683, 590.1916504, -1185.2105713, 1298.6289062

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6498601, upper bound: 886.6556708
time: 1.30 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6813299, upper bound: 886.6820305
time: 1.46 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -324.6807556, 519.9534912, -500.8932495, 812.2302856, -1136.9110107, 1020.8467407
1: -363.1039124, 515.1547241, -564.1883545, 811.1494751, -1174.2534180, 1079.3430176
2: -364.9215393, 510.3396912, -563.0802002, 801.6936646, -1166.6151123, 1073.4199219
3: -445.3571777, 593.4895020, -694.2907104, 930.0158081, -1375.3729248, 1287.7800293
4: -393.2788391, 583.7371826, -605.3632812, 917.6698608, -1310.2155762, 1189.1003418

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6469796, upper bound: 886.6363146
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6813090, upper bound: 886.6813090
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -492.3429565, 798.1012573, -500.8932495, 812.2302856, -1304.5732422, 1298.9945068
1: -554.5442505, 796.8940430, -564.1883545, 811.1494751, -1364.4223633, 1359.8352051
2: -553.4963379, 787.6232910, -563.0802002, 801.6936646, -1353.8116455, 1349.4210205
3: -682.4190674, 913.6898193, -694.2907104, 930.0158081, -1611.9841309, 1607.4830322
4: -595.0189209, 901.7127686, -605.3632812, 917.6698608, -1510.1049805, 1504.5849609

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6469796, upper bound: 886.6363146
time: 1.41 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6813090, upper bound: 886.6813090
time: 1.17 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -321.0200500, 523.3755493, -475.1667786, 768.9403687, -1089.9604492, 998.5422974
1: -359.9479980, 516.3449707, -537.9765015, 769.8582764, -1129.8062744, 1054.3212891
2: -361.5169373, 510.5056763, -534.3039551, 759.2224121, -1120.7390137, 1044.8095703
3: -441.4639587, 594.6976929, -662.4543457, 883.9057617, -1325.3695068, 1257.1519775
4: -390.8002319, 582.0920410, -575.2274780, 868.3067017, -1258.9444580, 1157.3192139

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6566470, upper bound: 886.6455236
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6566470, upper bound: 886.6537618
time: 1.46 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -328.5566406, 535.5642090, -474.4230347, 767.7185669, -1096.2751465, 1009.9872437
1: -368.4096375, 528.4622192, -537.1472168, 768.6757202, -1137.0853271, 1065.6093750
2: -369.9674683, 522.6040039, -533.4801025, 758.0422363, -1128.0095215, 1056.0841064
3: -451.7708130, 608.6307373, -661.3934937, 882.5667725, -1334.3374023, 1270.0239258
4: -399.7749023, 596.0436401, -574.4714355, 866.8731689, -1266.6479492, 1170.5150146

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6424470, upper bound: 886.6257089
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6508356, upper bound: 886.6388472
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -340.6291199, 555.3612671, -483.5437317, 782.1271973, -1122.7561035, 1038.9046631
1: -382.2075195, 547.5028076, -547.4036865, 781.0273438, -1163.2347412, 1094.5982666
2: -383.6327515, 541.5170288, -544.0492554, 770.6643066, -1154.2971191, 1085.5662842
3: -468.4717102, 630.6705322, -672.7966919, 897.1543579, -1365.6257324, 1303.4670410
4: -415.1465759, 617.6990356, -586.4437256, 881.8234253, -1296.4139404, 1204.1424561

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6693365, upper bound: 886.6662661
time: 1.30 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6693365, upper bound: 886.6662661
time: 1.17 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -357.3765259, 586.4533691, -466.0512695, 755.2816162, -1112.6582031, 1052.5046387
1: -400.0353088, 576.5250854, -527.1956787, 752.8840332, -1152.9191895, 1103.7207031
2: -402.3362122, 570.4312134, -524.3140259, 742.6513672, -1144.9875488, 1094.7452393
3: -490.3978271, 664.4187012, -647.8148193, 865.5985718, -1355.9963379, 1312.2335205
4: -433.0387573, 649.4725952, -564.3891602, 849.1672363, -1282.2059326, 1213.8616943

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6679562, upper bound: 886.6651721
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6679562, upper bound: 886.6651721
time: 1.24 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -343.3114014, 558.0192261, -488.2839966, 790.4736938, -1133.7850342, 1046.3031006
1: -384.5410767, 550.9949341, -552.6983032, 791.5245972, -1176.0656738, 1103.6932373
2: -385.8178711, 544.8323975, -548.9578857, 780.6649780, -1166.4829102, 1093.7900391
3: -472.1747131, 634.9761353, -680.5823364, 908.7113037, -1380.8859863, 1315.5582275
4: -416.0151672, 622.0160522, -590.7958374, 892.7746582, -1307.4278564, 1212.5620117

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6539115, upper bound: 886.6343081
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6603765, upper bound: 886.6481059
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -356.8437195, 584.2464600, -470.8910522, 763.9508667, -1120.7945557, 1055.1374512
1: -398.8213196, 574.8706665, -532.5656128, 763.5835571, -1162.4047852, 1107.4361572
2: -401.0133057, 568.6564331, -529.2709961, 752.8795776, -1153.8925781, 1097.9272461
3: -489.5665894, 662.9502563, -655.7095947, 877.4799805, -1367.0466309, 1318.6599121
4: -430.1664124, 647.9262085, -568.8646240, 860.3057861, -1290.4718018, 1216.7907715

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6517769, upper bound: 886.6337176
time: 1.46 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6568726, upper bound: 886.6468169
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -356.6286621, 579.4489136, -494.4740601, 800.3046265, -1156.9333496, 1073.9226074
1: -399.5567322, 571.8599854, -559.5927124, 799.4019775, -1198.9582520, 1131.4526367
2: -401.0542297, 565.6738892, -556.3513184, 788.8375244, -1189.8917236, 1122.0250244
3: -490.3020325, 659.0007935, -687.9149780, 918.2416992, -1408.5434570, 1346.9155273
4: -432.8078613, 645.9919434, -599.5607910, 902.3804321, -1334.4013672, 1245.4217529

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6701398, upper bound: 886.6662661
time: 1.11 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6701398, upper bound: 886.6662661
time: 1.28 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -370.6916809, 606.4636841, -476.6315002, 773.0581665, -1143.7496338, 1083.0952148
1: -414.4230042, 596.6043701, -538.9847412, 770.8088989, -1185.2319336, 1135.5891113
2: -416.6798401, 590.2903442, -536.2153931, 760.4130859, -1177.0928955, 1126.5057373
3: -508.4282227, 687.8913574, -662.4562378, 886.2391357, -1394.6672363, 1350.3475342
4: -447.4330750, 672.7003174, -577.0866089, 869.1971436, -1316.6300049, 1249.7868652

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6676328, upper bound: 886.6652772
time: 1.33 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6676328, upper bound: 886.6652772
time: 1.43 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -458.8282471, 742.7175293, -319.2762451, 521.8519897, -980.6801758, 1061.9937744
1: -519.8280029, 741.9848022, -358.7432861, 515.9518433, -1035.7797852, 1100.7280273
2: -516.4671021, 731.8739014, -359.3713074, 509.6116638, -1026.0787354, 1091.2451172
3: -638.9916382, 852.3137207, -440.3494568, 594.4904175, -1233.4819336, 1292.6629639
4: -557.2080078, 837.1256104, -388.8027649, 580.6963501, -1137.9042969, 1225.9283447

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6528749, upper bound: 886.6436201
time: 1.13 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6540936, upper bound: 886.6439151
time: 1.01 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -457.5454712, 740.6235962, -321.4477539, 525.0524292, -982.5979004, 1062.0711670
1: -518.4194946, 739.9350586, -361.0856323, 519.1340942, -1037.5533447, 1101.0205078
2: -515.0523071, 729.8735962, -361.7395935, 512.8002930, -1027.8524170, 1091.6131592
3: -637.1731567, 849.9403687, -443.2771301, 598.1270752, -1235.3002930, 1293.2172852
4: -555.8087158, 834.7437744, -391.2297058, 584.3536987, -1140.1623535, 1225.9735107

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6418294, upper bound: 886.6246964
time: 1.42 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6490312, upper bound: 886.6372205
time: 1.15 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -476.7803650, 771.6308594, -350.6332703, 569.7678833, -1046.5480957, 1122.2640381
1: -539.9763184, 772.4961548, -392.9486694, 562.4432373, -1102.4194336, 1165.4448242
2: -536.3087158, 761.8306885, -394.4375305, 556.1668701, -1092.4754639, 1156.2680664
3: -664.6658936, 886.8717041, -481.8733521, 648.0060425, -1312.6718750, 1368.7449951
4: -577.7337036, 871.2992554, -426.3060913, 635.1034546, -1212.8370361, 1297.6051025

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6395592, upper bound: 886.6387204
time: 1.39 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6409468, upper bound: 886.6598138
time: 1.16 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -480.2467346, 776.7702637, -350.6332703, 569.7678833, -1050.0146484, 1127.4034424
1: -543.8629150, 775.5766602, -392.9486694, 562.4432373, -1106.3061523, 1168.5252686
2: -540.5451050, 765.2640991, -394.4375305, 556.1668701, -1096.7116699, 1159.7015381
3: -668.1536255, 890.8625488, -481.8733521, 648.0060425, -1316.1596680, 1372.7358398
4: -583.1776733, 875.6425781, -426.3060913, 635.1034546, -1218.2811279, 1301.9486084

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6395592, upper bound: 886.6553281
time: 1.22 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6409468, upper bound: 886.6645851
time: 1.12 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -483.8485718, 783.6398926, -339.7637024, 554.1115112, -1037.9600830, 1123.4035645
1: -547.6936646, 784.4992065, -381.3984680, 548.3135986, -1096.0073242, 1165.8977051
2: -544.0818481, 773.7381592, -382.1158752, 541.7268677, -1085.8087158, 1155.8536377
3: -674.3672485, 900.7269897, -468.4788513, 631.6959229, -1306.0629883, 1369.2058105
4: -585.6768188, 884.7250366, -412.3956604, 617.8391724, -1203.5159912, 1297.1206055

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6540187, upper bound: 886.6580908
time: 1.35 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6589226, upper bound: 886.6598102
time: 1.20 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -490.4929810, 794.2290649, -339.7637024, 554.1115112, -1044.6044922, 1133.9927979
1: -555.0930786, 793.1872559, -381.3984680, 548.3135986, -1103.4067383, 1174.5856934
2: -552.0131836, 782.6972656, -382.1158752, 541.7268677, -1093.7398682, 1164.8131104
3: -682.3418579, 911.2104492, -468.4788513, 631.6959229, -1314.0378418, 1379.6893311
4: -595.0556030, 895.1871338, -412.3956604, 617.8391724, -1212.8947754, 1307.5827637

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6540187, upper bound: 886.6580908
time: 1.07 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6589226, upper bound: 886.6598102
time: 1.32 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.78 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -886.6752950, upper bound: 886.6773695
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -886.6755477, upper bound: 886.6742497
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -886.6752950, upper bound: 886.6773695
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -886.6752950, upper bound: 886.6742497
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -886.6821580, upper bound: 886.6802686
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -886.6821580, upper bound: 886.6802686
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -886.6821580, upper bound: 886.6802686
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -886.6821580, upper bound: 886.6802686
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -886.6498601, upper bound: 886.6556708
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -886.6813299, upper bound: 886.6820305
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -886.6498601, upper bound: 886.6556708
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -886.6813299, upper bound: 886.6820305
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -886.6469796, upper bound: 886.6363146
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -886.6813090, upper bound: 886.6813090
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -886.6469796, upper bound: 886.6363146
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -886.6813090, upper bound: 886.6813090
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -886.6566470, upper bound: 886.6455236
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -886.6566470, upper bound: 886.6537618
IS_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.78
Output dim: 0, lower bound: -886.6424470, upper bound: 886.6257089
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -886.6508356, upper bound: 886.6388472
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -886.6693365, upper bound: 886.6662661
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -886.6693365, upper bound: 886.6662661
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -886.6679562, upper bound: 886.6651721
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -886.6679562, upper bound: 886.6651721
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -886.6539115, upper bound: 886.6343081
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -886.6603765, upper bound: 886.6481059
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -886.6517769, upper bound: 886.6337176
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -886.6568726, upper bound: 886.6468169
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -886.6701398, upper bound: 886.6662661
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -886.6701398, upper bound: 886.6662661
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -886.6676328, upper bound: 886.6652772
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -886.6676328, upper bound: 886.6652772
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -886.6528749, upper bound: 886.6436201
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -886.6540936, upper bound: 886.6439151
IS_A2_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.78
Output dim: 0, lower bound: -886.6418294, upper bound: 886.6246964
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -886.6490312, upper bound: 886.6372205
IS_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.78
Output dim: 0, lower bound: -886.6395592, upper bound: 886.6387204
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -886.6409468, upper bound: 886.6598138
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -886.6395592, upper bound: 886.6553281
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -886.6409468, upper bound: 886.6645851
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -886.6540187, upper bound: 886.6580908
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -886.6589226, upper bound: 886.6598102
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -886.6540187, upper bound: 886.6580908
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.78
Output dim: 0, lower bound: -886.6589226, upper bound: 886.6598102
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.78
Output dim: 0, lower bound: -886.6537771, upper bound: 886.6712821
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.78
Output dim: 0, lower bound: -886.6629846, upper bound: 886.6719840
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.78
Output dim: 0, lower bound: -886.6598134, upper bound: 886.6524734
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.78
Output dim: 0, lower bound: -886.6587849, upper bound: 886.6525468
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.78
Output dim: 0, lower bound: -886.6516152, upper bound: 886.6610141
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.78
Output dim: 0, lower bound: -886.6516152, upper bound: 886.6647744
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.78
Output dim: 0, lower bound: -886.6509988, upper bound: 886.6589119
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.78
Output dim: 0, lower bound: -886.6589119, upper bound: 886.6594811
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.78
Output dim: 0, lower bound: -886.6594811, upper bound: 886.6642197
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.78
Output dim: 0, lower bound: -886.6594811, upper bound: 886.6647744
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=1024.72607421875
rel_dist={0: [-886.6885385920837, 886.6885385920837]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 1117.25 seconds
