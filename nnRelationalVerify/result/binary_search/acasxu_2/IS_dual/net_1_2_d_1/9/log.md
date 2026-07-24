## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_2.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 57.280903066


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-13.0451412, 53.5286636, -13.0451412, 53.5286636, -66.5738068, 66.5738068)
1: (-16.5472050, 60.5470352, -16.5472050, 60.5470352, -77.0942383, 77.0942383)
2: (-16.2069473, 60.5032959, -16.2069473, 60.5032959, -76.7102280, 76.7102280)
3: (-27.8260193, 64.2862701, -27.8260193, 64.2862701, -92.1122894, 92.1122894)
4: (-25.8074226, 62.3490105, -25.8074226, 62.3490105, -88.1564331, 88.1564331)

## BASE Result
execution time: IAR + LP analysis = 2.07 + 1.58 = 3.65 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -57.5687468, upper bound: 57.5687468


# Binary Search by BASE starts (time budget: 1196.35 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.2500000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=66.57380676269531
rel_dist={0: [-57.5687467976788, 57.5687467976788]}

## Binary search (step 1) starts
Candidate diff: 0.1250000


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=66.57380676269531
rel_dist={0: [-57.5687467976788, 57.5687467976788]}

## Binary search (step 2) starts
Candidate diff: 0.0625000


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=66.57380676269531
rel_dist={0: [-57.5686962838552, 57.5686962838552]}

## Binary search (step 3) starts
Candidate diff: 0.0312500


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=66.57380676269531
rel_dist={0: [-57.56848922332033, 57.56848922332034]}

## Binary search (step 4) starts
Candidate diff: 0.0156250


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=66.57380676269531
rel_dist={0: [-57.568351401632306, 57.5683514016323]}

## Binary search (step 5) starts
Candidate diff: 0.0078125


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0078125, mid=0.0078125, abs_max=66.57380676269531
rel_dist={0: [-57.568257500820835, 57.568257500820835]}

## Binary search (step 6) starts
Candidate diff: 0.0039062


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0039062, mid=0.0039062, abs_max=66.57380676269531
rel_dist={0: [-57.5682064165504, 57.56820641655041]}

## Binary search (step 7) starts
Candidate diff: 0.0019531


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0019531, mid=0.0019531, abs_max=66.57380676269531
rel_dist={0: [-57.568180214636605, 57.56818021463661]}

## Binary search (step 8) starts
Candidate diff: 0.0009766


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0009766, mid=0.0009766, abs_max=66.57380676269531
rel_dist={0: [-57.56816602745435, 57.56816602745435]}

## Binary search (step 9) starts
Candidate diff: 0.0004883


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0004883, mid=0.0004883, abs_max=66.57380676269531
rel_dist={0: [-57.568158614525274, 57.56815861452529]}

## Binary search (step 10) starts
Candidate diff: 0.0002441


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0002441, mid=0.0002441, abs_max=66.57380676269531
rel_dist={0: [-57.56815490806733, 57.56815490806733]}

## Binary search (step 11) starts
Candidate diff: 0.0001221


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0001221, mid=0.0001221, abs_max=66.57380676269531
rel_dist={0: [-57.56815305485149, 57.56815305485149]}

## Binary search (step 12) starts
Candidate diff: 0.0000610


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0000610, mid=0.0000610, abs_max=66.57380676269531
rel_dist={0: [-57.568152128269546, 57.568152128269546]}

## Binary search (step 13) starts
Candidate diff: 0.0000305


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000305, mid=0.0000305, abs_max=66.57380676269531
rel_dist={0: [-57.5681516650294, 57.568151665029404]}

## Binary search (step 14) starts
Candidate diff: 0.0000153


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000153, mid=0.0000153, abs_max=66.57380676269531
rel_dist={0: [-57.568151433506756, 57.568151433506756]}

## Binary search (step 15) starts
Candidate diff: 0.0000076


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000076, mid=0.0000076, abs_max=66.57380676269531
rel_dist={0: [-57.56815131792488, 57.56815131792487]}

## Binary search (step 16) starts
Candidate diff: 0.0000038


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000038, mid=0.0000038, abs_max=66.57380676269531
rel_dist={0: [-57.56815126044128, 57.56815126044128]}

## Binary search (step 17) starts
Candidate diff: 0.0000019


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000019, mid=0.0000019, abs_max=66.57380676269531
rel_dist={0: [-57.56815123432093, 57.56815126003005]}

## Binary search (step 18) starts
Candidate diff: 0.0000010


## IAR start
Binary search (step 18): status=Status.UNKNOWN, low=0.0000000, high=0.0000010, mid=0.0000010, abs_max=66.57380676269531
rel_dist={0: [-57.568151283478784, 57.568151286271956]}

## Binary Search Result
Binary search time: 71.30 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 1125.05 seconds

## Binary search (step 0) starts
Candidate diff: 0.2500000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5573855, upper bound: 57.5408885
time: 0.54 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5684509, upper bound: 57.5684509
time: 1.32 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.05 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 2.05
Output dim: 0, lower bound: -57.5573855, upper bound: 57.5408885
IS_B2, status: Status.UNKNOWN, split count: 1, time: 2.05
Output dim: 0, lower bound: -57.5684509, upper bound: 57.5684509

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -12.7335720, 52.3666992, -11.9564142, 50.0494728, -62.7830315, 64.3231125
1: -16.1586933, 59.2372627, -15.2706614, 56.6176529, -72.7763214, 74.5079269
2: -15.8260145, 59.1612206, -14.8593369, 56.5337257, -72.3597412, 74.0205536
3: -27.1937332, 62.9044304, -25.9164085, 60.1328697, -87.3265991, 88.8208313
4: -25.2274284, 60.9320183, -23.8998127, 58.1139297, -83.3413544, 84.8318329

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_B1

### Relational analysis result of IS_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5502469, upper bound: 57.5350720
time: 0.53 seconds

## Relational analysis of IS_B1_B2

### Relational analysis result of IS_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4527388, upper bound: 57.3092460
time: 0.69 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -13.0451412, 53.5286636, -12.6128531, 51.9863625, -65.0315018, 66.1415176
1: -16.5472050, 60.5470352, -16.0097351, 58.8065300, -75.3537369, 76.5567703
2: -16.2069473, 60.5032959, -15.6801224, 58.7119789, -74.9189224, 76.1834106
3: -27.8260193, 64.2862701, -26.9607754, 62.4464188, -90.2724304, 91.2470398
4: -25.8074226, 62.3490105, -25.0127659, 60.4369431, -86.2443695, 87.3617630

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B2_B1

### Relational analysis result of IS_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5526995, upper bound: 57.5589585
time: 0.63 seconds

## Relational analysis of IS_B2_B2

### Relational analysis result of IS_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.57 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.40 seconds
IS_B1_B1, status: Status.UNKNOWN, split count: 2, time: 3.40
Output dim: 0, lower bound: -57.5502469, upper bound: 57.5350720
IS_B1_B2, status: Status.UNKNOWN, split count: 2, time: 3.40
Output dim: 0, lower bound: -57.4527388, upper bound: 57.3092460
IS_B2_B1, status: Status.UNKNOWN, split count: 2, time: 3.40
Output dim: 0, lower bound: -57.5526995, upper bound: 57.5589585
IS_B2_B2, status: Status.UNKNOWN, split count: 2, time: 3.40
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139

## BFS IS instance: IS_B1_B1

### Backsubstitution after applying IS history:
0: -12.5247002, 51.5509186, -9.2238035, 39.4067192, -51.9314194, 60.7747116
1: -15.8958445, 58.3167496, -11.8112040, 44.6088829, -60.5047264, 70.1279526
2: -15.5717020, 58.2214890, -11.5591335, 44.2477875, -59.8194771, 69.7806244
3: -26.7591343, 61.9355850, -20.2589512, 47.5700760, -74.3292084, 82.1945343
4: -24.8315468, 59.9630852, -18.8161907, 45.5512772, -70.3828125, 78.7792664

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B1_B1_A1

### Relational analysis result of IS_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5044373, upper bound: 57.5108207
time: 0.52 seconds

## Relational analysis of IS_B1_B1_A2

### Relational analysis result of IS_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5322142, upper bound: 57.5199219
time: 0.54 seconds

## BFS IS instance: IS_B1_B2

### Backsubstitution after applying IS history:
0: -12.7335720, 52.3666992, -11.1930752, 47.0669174, -59.8004761, 63.5597763
1: -16.1586933, 59.2372627, -14.3044930, 53.2331352, -69.3918304, 73.5417557
2: -15.8260145, 59.1612206, -13.9323788, 53.1038742, -68.9298859, 73.0935974
3: -27.1937332, 62.9044304, -24.3264256, 56.5921440, -83.7858734, 87.2308578
4: -25.2274284, 60.9320183, -22.4652710, 54.6005325, -79.8279572, 83.3972931

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_B2_A1

### Relational analysis result of IS_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4469725, upper bound: 57.3072592
time: 0.54 seconds

## Relational analysis of IS_B1_B2_A2

### Relational analysis result of IS_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4469725, upper bound: 57.3092460
time: 0.56 seconds

## BFS IS instance: IS_B2_B1

### Backsubstitution after applying IS history:
0: -13.0451412, 53.5286636, -10.6547174, 44.4431190, -57.4882584, 64.1833725
1: -16.5472050, 60.5470352, -13.5761242, 50.2670593, -66.8142624, 74.1231613
2: -16.2069473, 60.5032959, -13.2904520, 50.0807762, -66.2877121, 73.7937393
3: -27.8260193, 64.2862701, -22.9239864, 53.4544601, -81.2804718, 87.2102509
4: -25.8074226, 62.3490105, -21.2236214, 51.5289040, -77.3363266, 83.5726318

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_B1_B1

### Relational analysis result of IS_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5526995, upper bound: 57.5589585
time: 0.53 seconds

## Relational analysis of IS_B2_B1_B2

### Relational analysis result of IS_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5504415, upper bound: 57.5546539
time: 0.54 seconds

## BFS IS instance: IS_B2_B2

### Backsubstitution after applying IS history:
0: -12.9743776, 53.2614937, -14.5608664, 59.0598717, -72.0342484, 67.8223572
1: -16.4590034, 60.2458076, -18.4390030, 66.7630920, -83.2220917, 78.6848145
2: -16.1219673, 60.1950493, -18.1132774, 66.8004532, -82.9224243, 78.3083191
3: -27.6791096, 63.9695549, -30.9053497, 70.9381409, -98.6172409, 94.8748779
4: -25.6757088, 62.0266266, -28.6642246, 69.0792694, -94.7549744, 90.6908417

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_B2_B1

### Relational analysis result of IS_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5439184, upper bound: 57.5366463
time: 0.52 seconds

## Relational analysis of IS_B2_B2_B2

### Relational analysis result of IS_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.50 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.28 seconds
IS_B1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 0, lower bound: -57.5044373, upper bound: 57.5108207
IS_B1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 0, lower bound: -57.5322142, upper bound: 57.5199219
IS_B1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 0, lower bound: -57.4469725, upper bound: 57.3072592
IS_B1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 0, lower bound: -57.4469725, upper bound: 57.3092460
IS_B2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 0, lower bound: -57.5526995, upper bound: 57.5589585
IS_B2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 0, lower bound: -57.5504415, upper bound: 57.5546539
IS_B2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 0, lower bound: -57.5439184, upper bound: 57.5366463
IS_B2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139

## BFS IS instance: IS_B1_B1_A1

### Backsubstitution after applying IS history:
0: -9.2376308, 39.9100113, -9.2238035, 39.4067192, -48.6443443, 49.1338158
1: -11.7768116, 45.1596451, -11.8112040, 44.6088829, -56.3856964, 56.9708443
2: -11.6069050, 44.7116089, -11.5591335, 44.2477875, -55.8546867, 56.2707405
3: -20.1845932, 48.1117172, -20.2589512, 47.5700760, -67.7546692, 68.3706589
4: -18.8402061, 45.8996506, -18.8161907, 45.5512772, -64.3914642, 64.7158356

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_B1_A1_A1

### Relational analysis result of IS_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4892642, upper bound: 57.5051435
time: 0.58 seconds

## Relational analysis of IS_B1_B1_A1_A2

### Relational analysis result of IS_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4892642, upper bound: 57.5108207
time: 0.60 seconds

## BFS IS instance: IS_B1_B1_A2

### Backsubstitution after applying IS history:
0: -10.2778044, 42.9593430, -9.0251799, 38.6557541, -48.9335594, 51.9845238
1: -13.0889339, 48.6212959, -11.5584135, 43.7627220, -56.8516541, 60.1797104
2: -12.8462124, 48.3089104, -11.3209457, 43.3808517, -56.2270660, 59.6298561
3: -22.0906181, 51.8160248, -19.8415718, 46.6844025, -68.7750092, 71.6575851
4: -20.5009594, 49.6421356, -18.4466991, 44.6605988, -65.1615601, 68.0888290

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_B1_A2_A1

### Relational analysis result of IS_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5271606, upper bound: 57.5182206
time: 0.54 seconds

## Relational analysis of IS_B1_B1_A2_A2

### Relational analysis result of IS_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5271606, upper bound: 57.5199219
time: 0.55 seconds

## BFS IS instance: IS_B1_B2_A1

### Backsubstitution after applying IS history:
0: -9.9311981, 41.5068092, -11.1930752, 47.0669174, -56.9981155, 52.6998825
1: -12.6394196, 46.9929504, -14.3044930, 53.2331352, -65.8725586, 61.2974434
2: -12.4209604, 46.6616096, -13.9323788, 53.1038742, -65.5248337, 60.5939865
3: -21.4286194, 50.1027603, -24.3264256, 56.5921440, -78.0207672, 74.4291840
4: -19.9921494, 48.0295029, -22.4652710, 54.6005325, -74.5926819, 70.4947662

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B1_B2_A1_B1

### Relational analysis result of IS_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4178729, upper bound: 57.2704551
time: 0.50 seconds

## Relational analysis of IS_B1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B1_B2_A1_A1

### Relational analysis result of IS_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -57.2045298, upper bound: 57.2045298
time: 0.52 seconds

## Relational analysis of IS_B1_B2_A1_A2

### Relational analysis result of IS_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2045298, upper bound: 57.3072592
time: 0.59 seconds

## BFS IS instance: IS_B1_B2_A2

### Backsubstitution after applying IS history:
0: -11.7358122, 48.5500641, -11.1930752, 47.0669174, -58.8027191, 59.7431412
1: -14.9050369, 54.9172630, -14.3044930, 53.2331352, -68.1381683, 69.2217560
2: -14.6121826, 54.7730751, -13.9323788, 53.1038742, -67.7160568, 68.7054520
3: -25.1381531, 58.3638763, -24.3264256, 56.5921440, -81.7302933, 82.6903000
4: -23.3387756, 56.4076157, -22.4652710, 54.6005325, -77.9393082, 78.8728867

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B1_B2_A2_B1

### Relational analysis result of IS_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4178729, upper bound: 57.2721923
time: 0.58 seconds

## Relational analysis of IS_B1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B1_B2_A2_A1

### Relational analysis result of IS_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -57.2045298, upper bound: 57.2045298
time: 0.92 seconds

## Relational analysis of IS_B1_B2_A2_A2

### Relational analysis result of IS_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2045298, upper bound: 57.3092460
time: 0.57 seconds

## BFS IS instance: IS_B2_B1_B1

### Backsubstitution after applying IS history:
0: -13.0451412, 53.5286636, -8.8559608, 37.6358414, -50.6809845, 62.3846092
1: -16.5472050, 60.5470352, -11.3271160, 42.6399384, -59.1871300, 71.8741455
2: -16.2069473, 60.5032959, -11.1272030, 42.3047295, -58.5116692, 71.6305008
3: -27.8260193, 64.2862701, -19.2453804, 45.3962097, -73.2222137, 83.5316467
4: -25.8074226, 62.3490105, -17.8188610, 43.6135788, -69.4210052, 80.1678543

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_B1_B1_B1

### Relational analysis result of IS_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5055907, upper bound: 57.5283342
time: 0.57 seconds

## Relational analysis of IS_B2_B1_B1_B2

### Relational analysis result of IS_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5097612, upper bound: 57.5387190
time: 0.53 seconds

## BFS IS instance: IS_B2_B1_B2

### Backsubstitution after applying IS history:
0: -13.0451412, 53.5286636, -9.9143133, 41.6221619, -54.6673050, 63.4429665
1: -16.5472050, 60.5470352, -12.6455078, 47.0909157, -63.6381226, 73.1925354
2: -16.2069473, 60.5032959, -12.3921022, 46.8379173, -63.0448608, 72.8953781
3: -27.8260193, 64.2862701, -21.4043350, 50.1154404, -77.9414597, 85.6906052
4: -25.8074226, 62.3490105, -19.8197823, 48.2272835, -74.0347061, 82.1687927

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_B1_B2_B1

### Relational analysis result of IS_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5060247, upper bound: 57.5292730
time: 0.51 seconds

## Relational analysis of IS_B2_B1_B2_B2

### Relational analysis result of IS_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5068245, upper bound: 57.5320031
time: 0.75 seconds

## BFS IS instance: IS_B2_B2_B1

### Backsubstitution after applying IS history:
0: -12.9743776, 53.2614937, -11.8374672, 48.4148979, -61.3892632, 65.0989609
1: -16.4590034, 60.2458076, -15.0364037, 54.8208542, -71.2798462, 75.2822037
2: -16.1219673, 60.1950493, -14.8005733, 54.6966438, -70.8186111, 74.9956207
3: -27.6791096, 63.9695549, -25.2378845, 58.3669090, -86.0460205, 89.2074356
4: -25.6757088, 62.0266266, -23.5150375, 56.6646614, -82.3403549, 85.5416641

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_B2_B1_A1

### Relational analysis result of IS_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4920778, upper bound: 57.5125861
time: 0.53 seconds

## Relational analysis of IS_B2_B2_B1_A2

### Relational analysis result of IS_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5277472, upper bound: 57.5245366
time: 0.55 seconds

## BFS IS instance: IS_B2_B2_B2

### Backsubstitution after applying IS history:
0: -12.9743776, 53.2614937, -12.4338989, 51.4096909, -64.3840714, 65.6953888
1: -16.4590034, 60.2458076, -15.7874660, 58.1604767, -74.6194763, 76.0332642
2: -16.1219673, 60.1950493, -15.5188274, 57.9909248, -74.1128922, 75.7138748
3: -27.6791096, 63.9695549, -26.5944901, 61.8139153, -89.4930267, 90.5640411
4: -25.6757088, 62.0266266, -24.7338448, 59.7358131, -85.4115219, 86.7604675

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_B2_B2_A1

### Relational analysis result of IS_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.61 seconds

## Relational analysis of IS_B2_B2_B2_A2

### Relational analysis result of IS_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
time: 0.55 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.39 seconds
IS_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 3.39
Output dim: 0, lower bound: -57.4892642, upper bound: 57.5051435
IS_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.39
Output dim: 0, lower bound: -57.4892642, upper bound: 57.5108207
IS_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.39
Output dim: 0, lower bound: -57.5271606, upper bound: 57.5182206
IS_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.39
Output dim: 0, lower bound: -57.5271606, upper bound: 57.5199219
IS_B1_B2_A1_A1, status: Status.VERIFIED, split count: 4, time: 3.39
Output dim: 0, lower bound: -57.2045298, upper bound: 57.2045298
IS_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.39
Output dim: 0, lower bound: -57.2045298, upper bound: 57.3072592
IS_B1_B2_A2_A1, status: Status.VERIFIED, split count: 4, time: 3.39
Output dim: 0, lower bound: -57.2045298, upper bound: 57.2045298
IS_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.39
Output dim: 0, lower bound: -57.2045298, upper bound: 57.3092460
IS_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.39
Output dim: 0, lower bound: -57.5055907, upper bound: 57.5283342
IS_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.39
Output dim: 0, lower bound: -57.5097612, upper bound: 57.5387190
IS_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.39
Output dim: 0, lower bound: -57.5060247, upper bound: 57.5292730
IS_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.39
Output dim: 0, lower bound: -57.5068245, upper bound: 57.5320031
IS_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.39
Output dim: 0, lower bound: -57.4920778, upper bound: 57.5125861
IS_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.39
Output dim: 0, lower bound: -57.5277472, upper bound: 57.5245366
IS_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.39
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139
IS_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.39
Output dim: 0, lower bound: -57.5487139, upper bound: 57.5487139

## BFS IS instance: IS_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -6.8784938, 30.7990665, -9.2238035, 39.4067192, -46.2852135, 40.0228691
1: -8.7717276, 34.9791870, -11.8112040, 44.6088829, -53.3806114, 46.7903862
2: -8.7575617, 34.2116394, -11.5591335, 44.2477875, -53.0053482, 45.7707748
3: -15.2604456, 37.5171890, -20.2589512, 47.5700760, -62.8305206, 57.7761345
4: -14.5341740, 35.1157303, -18.8161907, 45.5512772, -60.0854378, 53.9319229

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B1_B1_A1_A1_A1

### Relational analysis result of IS_B1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2659297, upper bound: 57.4044075
time: 0.54 seconds

## Relational analysis of IS_B1_B1_A1_A1_A2

### Relational analysis result of IS_B1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2659297, upper bound: 57.5051435
time: 0.60 seconds

## BFS IS instance: IS_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -8.6025705, 37.5422592, -9.2238035, 39.4067192, -48.0092850, 46.7660637
1: -10.9746580, 42.4869690, -11.8112040, 44.6088829, -55.5835419, 54.2981720
2: -10.8450823, 41.9903107, -11.5591335, 44.2477875, -55.0928688, 53.5494461
3: -18.8741951, 45.3108101, -20.2589512, 47.5700760, -66.4442596, 65.5697632
4: -17.6569481, 43.1126709, -18.8161907, 45.5512772, -63.2082062, 61.9288521

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B1_B1_A1_A2_A1

### Relational analysis result of IS_B1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2659297, upper bound: 57.4044075
time: 0.53 seconds

## Relational analysis of IS_B1_B1_A1_A2_A2

### Relational analysis result of IS_B1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2659297, upper bound: 57.5108207
time: 0.56 seconds

## BFS IS instance: IS_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -8.2486143, 35.2901955, -9.0251799, 38.6557541, -46.9043617, 44.3153725
1: -10.5443096, 40.0452499, -11.5584135, 43.7627220, -54.3070297, 51.6036644
2: -10.4107323, 39.4347649, -11.3209457, 43.3808517, -53.7915840, 50.7557106
3: -17.9483242, 42.8938828, -19.8415718, 46.6844025, -64.6327209, 62.7354546
4: -16.8446903, 40.5732803, -18.4466991, 44.6605988, -61.5052757, 59.0199814

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B1_B1_A2_A1_A1

### Relational analysis result of IS_B1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5031397, upper bound: 57.4526942
time: 0.56 seconds

## Relational analysis of IS_B1_B1_A2_A1_A2

### Relational analysis result of IS_B1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5031905, upper bound: 57.4529045
time: 0.63 seconds

## BFS IS instance: IS_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -9.6735115, 40.6550446, -9.0251799, 38.6557541, -48.3292656, 49.6802254
1: -12.3292837, 46.0220909, -11.5584135, 43.7627220, -56.0920029, 57.5805054
2: -12.1157894, 45.6436424, -11.3209457, 43.3808517, -55.4966431, 56.9645882
3: -20.8493938, 49.0880966, -19.8415718, 46.6844025, -67.5337830, 68.9296570
4: -19.3699036, 46.9194641, -18.4466991, 44.6605988, -64.0304947, 65.3661575

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_B1_A2_A2_B1

### Relational analysis result of IS_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5256938, upper bound: 57.5110002
time: 0.54 seconds

## Relational analysis of IS_B1_B1_A2_A2_B2

### Relational analysis result of IS_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5256938, upper bound: 57.5199219
time: 0.63 seconds

## BFS IS instance: IS_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -9.8234758, 41.2086678, -11.1930752, 47.0669174, -56.8903885, 52.4017372
1: -12.5056238, 46.6563835, -14.3044930, 53.2331352, -65.7387543, 60.9608765
2: -12.2946529, 46.3034515, -13.9323788, 53.1038742, -65.3985291, 60.2358322
3: -21.2287254, 49.7448921, -24.3264256, 56.5921440, -77.8208694, 74.0713196
4: -19.8178635, 47.6446228, -22.4652710, 54.6005325, -74.4183960, 70.1098938

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B1_B2_A1_A2_B1

### Relational analysis result of IS_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4044075, upper bound: 57.2704551
time: 0.55 seconds

## Relational analysis of IS_B1_B2_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B1_B2_A1_A2_B1

### Relational analysis result of IS_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3407174, upper bound: 57.1368771
time: 0.57 seconds

## Relational analysis of IS_B1_B2_A1_A2_B2

### Relational analysis result of IS_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4245223, upper bound: 57.2982279
time: 0.60 seconds

## BFS IS instance: IS_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -11.5943317, 48.1005936, -11.1930752, 47.0669174, -58.6612473, 59.2936707
1: -14.7288771, 54.4083939, -14.3044930, 53.2331352, -67.9620056, 68.7128906
2: -14.4410677, 54.2435684, -13.9323788, 53.1038742, -67.5449448, 68.1759491
3: -24.8644619, 57.8241615, -24.3264256, 56.5921440, -81.4566040, 82.1505890
4: -23.0881882, 55.8258858, -22.4652710, 54.6005325, -77.6887207, 78.2911530

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B1_B2_A2_A2_B1

### Relational analysis result of IS_B1_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -57.0199985, upper bound: 57.1046118
time: 0.57 seconds

## Relational analysis of IS_B1_B2_A2_A2_B2

### Relational analysis result of IS_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.1950829, upper bound: 57.2870526
time: 0.58 seconds

## BFS IS instance: IS_B2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -12.8046780, 52.5998383, -6.4694872, 28.4099197, -41.2145958, 59.0693245
1: -16.2452068, 59.4991722, -8.3112011, 32.2974014, -48.5426102, 67.8103714
2: -15.9143915, 59.4335175, -8.2349968, 31.6430893, -47.5574799, 67.6685104
3: -27.3275928, 63.1831360, -14.2844219, 34.6641693, -61.9917603, 77.4675598
4: -25.3546982, 61.2416534, -13.5163860, 32.6297150, -57.9844131, 74.7580414

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_B1_B1_B1_A1

### Relational analysis result of IS_B2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4781464, upper bound: 57.5070317
time: 0.60 seconds

## Relational analysis of IS_B2_B1_B1_B1_A2

### Relational analysis result of IS_B2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4709592, upper bound: 57.5040013
time: 2.03 seconds

## BFS IS instance: IS_B2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -13.0451412, 53.5286636, -8.1467390, 35.0198555, -48.0649948, 61.6753998
1: -16.5472050, 60.5470352, -10.4393291, 39.6870842, -56.2342873, 70.9863586
2: -16.2069473, 60.5032959, -10.2659464, 39.2754898, -55.4824371, 70.7692337
3: -27.8260193, 64.2862701, -17.7922592, 42.2942581, -70.1202774, 82.0785294
4: -25.8074226, 62.3490105, -16.5044823, 40.4799461, -66.2873688, 78.8534927

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_B1_B1_B2_A1

### Relational analysis result of IS_B2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5097612, upper bound: 57.5387190
time: 0.55 seconds

## Relational analysis of IS_B2_B1_B1_B2_A2

### Relational analysis result of IS_B2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5097612, upper bound: 57.5387190
time: 0.54 seconds

## BFS IS instance: IS_B2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -12.8046780, 52.5998383, -7.1907134, 31.1665955, -43.9712715, 59.7905502
1: -16.2452068, 59.4991722, -9.2092686, 35.3581696, -51.6033783, 68.7084351
2: -15.9143915, 59.4335175, -9.1051140, 34.7669830, -50.6813698, 68.5386353
3: -27.3275928, 63.1831360, -15.7794971, 37.8664360, -65.1940308, 78.9626312
4: -25.3546982, 61.2416534, -14.8050480, 35.8460159, -61.2007141, 76.0466843

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_B1_B2_B1_A1

### Relational analysis result of IS_B2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5060247, upper bound: 57.5292730
time: 0.55 seconds

## Relational analysis of IS_B2_B1_B2_B1_A2

### Relational analysis result of IS_B2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5060247, upper bound: 57.5292730
time: 0.59 seconds

## BFS IS instance: IS_B2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -13.0451412, 53.5286636, -8.9674406, 38.0639191, -51.1090622, 62.4960938
1: -16.5472050, 60.5470352, -11.4526777, 43.0664177, -59.6136055, 71.9997101
2: -16.2069473, 60.5032959, -11.2560825, 42.7363510, -58.9432907, 71.7593689
3: -27.8260193, 64.2862701, -19.4629898, 45.9011917, -73.7271957, 83.7492599
4: -25.8074226, 62.3490105, -18.0676479, 44.0224457, -69.8298645, 80.4166412

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_B1_B2_B2_A1

### Relational analysis result of IS_B2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5068245, upper bound: 57.5320031
time: 0.58 seconds

## Relational analysis of IS_B2_B1_B2_B2_A2

### Relational analysis result of IS_B2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5068245, upper bound: 57.5320031
time: 0.53 seconds

## BFS IS instance: IS_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -9.6865273, 41.6282005, -11.8374672, 48.4148979, -58.1014252, 53.4656563
1: -12.3476610, 47.0912514, -15.0364037, 54.8208542, -67.1685104, 62.1276550
2: -12.1525879, 46.6972885, -14.8005733, 54.6966438, -66.8492279, 61.4978561
3: -21.1138229, 50.1320953, -25.2378845, 58.3669090, -79.4807281, 75.3699646
4: -19.6613407, 47.9430275, -23.5150375, 56.6646614, -76.3259964, 71.4580688

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_B2_B1_A1_A1

### Relational analysis result of IS_B2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4920778, upper bound: 57.5125861
time: 0.55 seconds

## Relational analysis of IS_B2_B2_B1_A1_A2

### Relational analysis result of IS_B2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4920778, upper bound: 57.5125861
time: 0.53 seconds

## BFS IS instance: IS_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -11.3257437, 46.7511139, -11.5214596, 47.2093582, -58.5351028, 58.2725716
1: -14.4109058, 52.8788795, -14.6393099, 53.4600296, -67.8709183, 67.5181885
2: -14.1109343, 52.6994209, -14.4142208, 53.3048897, -67.4158249, 67.1136322
3: -24.2345390, 56.3043213, -24.5794067, 56.9279633, -81.1624832, 80.8837280
4: -22.3835793, 54.1908875, -22.9001389, 55.2124405, -77.5960083, 77.0910110

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_B2_B1_A2_A1

### Relational analysis result of IS_B2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5209701, upper bound: 57.5209702
time: 0.58 seconds

## Relational analysis of IS_B2_B2_B1_A2_A2

### Relational analysis result of IS_B2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5209701, upper bound: 57.5245366
time: 0.64 seconds

## BFS IS instance: IS_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -11.0797024, 45.9357185, -12.4338989, 51.4096909, -62.4893951, 58.3696175
1: -14.1072273, 51.9529762, -15.7874660, 58.1604767, -72.2677002, 67.7404251
2: -13.8058453, 51.8173180, -15.5188274, 57.9909248, -71.7967682, 67.3361359
3: -23.7680473, 55.2296677, -26.5944901, 61.8139153, -85.5819626, 81.8241425
4: -21.9876976, 53.3685150, -24.7338448, 59.7358131, -81.7235107, 78.1023560

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B2_B2_B2_A1_B1

### Relational analysis result of IS_B2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5193631, upper bound: 57.4956442
time: 0.61 seconds

## Relational analysis of IS_B2_B2_B2_A1_B2

### Relational analysis result of IS_B2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5313136, upper bound: 57.5313135
time: 0.54 seconds

## BFS IS instance: IS_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -14.9150190, 60.2439690, -12.4338989, 51.4096909, -66.3247070, 72.6778564
1: -18.8722439, 68.0997391, -15.7874660, 58.1604767, -77.0327225, 83.8871918
2: -18.5386639, 68.1810379, -15.5188274, 57.9909248, -76.5295868, 83.6998520
3: -31.5919933, 72.3570404, -26.5944901, 61.8139153, -93.4059067, 98.9515152
4: -29.2980194, 70.5712891, -24.7338448, 59.7358131, -89.0338287, 95.3051300

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_B2_B2_A2_A1

### Relational analysis result of IS_B2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5318508, upper bound: 57.5439184
time: 0.55 seconds

## Relational analysis of IS_B2_B2_B2_A2_A2

### Relational analysis result of IS_B2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5318508, upper bound: 57.5487139
time: 0.60 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.42 seconds
IS_B1_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -57.2659297, upper bound: 57.4044075
IS_B1_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -57.2659297, upper bound: 57.5051435
IS_B1_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -57.2659297, upper bound: 57.4044075
IS_B1_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -57.2659297, upper bound: 57.5108207
IS_B1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -57.5031397, upper bound: 57.4526942
IS_B1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -57.5031905, upper bound: 57.4529045
IS_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -57.5256938, upper bound: 57.5110002
IS_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -57.5256938, upper bound: 57.5199219
IS_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -57.3407174, upper bound: 57.1368771
IS_B1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -57.4245223, upper bound: 57.2982279
IS_B1_B2_A2_A2_B1, status: Status.VERIFIED, split count: 5, time: 3.42
Output dim: 0, lower bound: -57.0199985, upper bound: 57.1046118
IS_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -57.1950829, upper bound: 57.2870526
IS_B2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -57.4781464, upper bound: 57.5070317
IS_B2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -57.4709592, upper bound: 57.5040013
IS_B2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -57.5097612, upper bound: 57.5387190
IS_B2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -57.5097612, upper bound: 57.5387190
IS_B2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -57.5060247, upper bound: 57.5292730
IS_B2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -57.5060247, upper bound: 57.5292730
IS_B2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -57.5068245, upper bound: 57.5320031
IS_B2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -57.5068245, upper bound: 57.5320031
IS_B2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -57.4920778, upper bound: 57.5125861
IS_B2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -57.4920778, upper bound: 57.5125861
IS_B2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -57.5209701, upper bound: 57.5209702
IS_B2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -57.5209701, upper bound: 57.5245366
IS_B2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -57.5193631, upper bound: 57.4956442
IS_B2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -57.5313136, upper bound: 57.5313135
IS_B2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -57.5318508, upper bound: 57.5439184
IS_B2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.42
Output dim: 0, lower bound: -57.5318508, upper bound: 57.5487139

## BFS IS instance: IS_B1_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: -6.0675859, 28.3898735, -9.2238035, 39.4067192, -45.4743042, 37.6136780
1: -7.7485080, 32.3027954, -11.8112040, 44.6088829, -52.3573837, 44.1139984
2: -7.7708387, 31.4510880, -11.5591335, 44.2477875, -52.0186272, 43.0102234
3: -13.7540531, 34.6235008, -20.2589512, 47.5700760, -61.3241272, 54.8824425
4: -13.1703291, 32.2367897, -18.8161907, 45.5512772, -58.7216072, 51.0529785

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_B1_A1_A1_A1_B1

### Relational analysis result of IS_B1_B1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4642506, upper bound: 57.4896601
time: 0.57 seconds

## Relational analysis of IS_B1_B1_A1_A1_A1_B2

### Relational analysis result of IS_B1_B1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4657174, upper bound: 57.4969617
time: 0.55 seconds

## BFS IS instance: IS_B1_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -6.7752047, 30.5335026, -9.2238035, 39.4067192, -46.1819191, 39.7573051
1: -8.6345415, 34.6835976, -11.8112040, 44.6088829, -53.2434235, 46.4948006
2: -8.6368971, 33.8876228, -11.5591335, 44.2477875, -52.8846855, 45.4467545
3: -15.0551920, 37.1991501, -20.2589512, 47.5700760, -62.6252670, 57.4580994
4: -14.3692780, 34.7664604, -18.8161907, 45.5512772, -59.9205360, 53.5826492

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_B1_A1_A1_A2_A1

### Relational analysis result of IS_B1_B1_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4417602, upper bound: 57.4874880
time: 0.53 seconds

## Relational analysis of IS_B1_B1_A1_A1_A2_A2

### Relational analysis result of IS_B1_B1_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4657174, upper bound: 57.5051435
time: 0.54 seconds

## BFS IS instance: IS_B1_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -7.9782028, 35.9070625, -9.2238035, 39.4067192, -47.3849220, 45.1308670
1: -10.2312508, 40.6491737, -11.8112040, 44.6088829, -54.8401337, 52.4603767
2: -10.0926924, 40.1304741, -11.5591335, 44.2477875, -54.3404808, 51.6895981
3: -17.8641090, 43.3818817, -20.2589512, 47.5700760, -65.4341888, 63.6408272
4: -16.7275486, 41.1993408, -18.8161907, 45.5512772, -62.2788239, 60.0155334

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_B1_A1_A2_A1_B1

### Relational analysis result of IS_B1_B1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2644628, upper bound: 57.3971059
time: 0.57 seconds

## Relational analysis of IS_B1_B1_A1_A2_A1_B2

### Relational analysis result of IS_B1_B1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2659297, upper bound: 57.4044075
time: 0.74 seconds

## BFS IS instance: IS_B1_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -8.4552927, 37.0844460, -9.2238035, 39.4067192, -47.8620071, 46.3082466
1: -10.7860126, 41.9743881, -11.8112040, 44.6088829, -55.3948975, 53.7855911
2: -10.6693573, 41.4484825, -11.5591335, 44.2477875, -54.9171410, 53.0076103
3: -18.5805569, 44.7684402, -20.2589512, 47.5700760, -66.1506271, 65.0273895
4: -17.4033432, 42.5424194, -18.8161907, 45.5512772, -62.9545898, 61.3586121

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_B1_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_B1_A1_A2_A2_B1

### Relational analysis result of IS_B1_B1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2644628, upper bound: 57.4994701
time: 0.54 seconds

## Relational analysis of IS_B1_B1_A1_A2_A2_B2

### Relational analysis result of IS_B1_B1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2659297, upper bound: 57.5108207
time: 0.58 seconds

## BFS IS instance: IS_B1_B1_A2_A1_A1

### Backsubstitution after applying IS history:
0: -5.8696084, 26.5953636, -8.8738098, 38.0590744, -43.9286804, 35.4691734
1: -7.4998922, 30.3397484, -11.3669062, 43.0893440, -50.5892372, 41.7066498
2: -7.5462909, 29.3994694, -11.1374245, 42.6968994, -50.2431831, 40.5368919
3: -12.9849119, 32.7016525, -19.5241280, 45.9783096, -58.9632225, 52.2257805
4: -12.6711483, 30.1201725, -18.1578388, 43.9637909, -56.6349411, 48.2780037

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_B1_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_B1_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B1_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_B1_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B1_B1_A2_A1_A1_A1

### Relational analysis result of IS_B1_B1_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4897438, upper bound: 57.4483449
time: 0.59 seconds

## Relational analysis of IS_B1_B1_A2_A1_A1_A2

### Relational analysis result of IS_B1_B1_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4897438, upper bound: 57.4527688
time: 0.52 seconds

## BFS IS instance: IS_B1_B1_A2_A1_A2

### Backsubstitution after applying IS history:
0: -7.9569707, 34.0704956, -9.0251799, 38.6557541, -46.6127243, 43.0956688
1: -10.1826277, 38.6770706, -11.5584135, 43.7627220, -53.9453506, 50.2354851
2: -10.0612640, 38.0595512, -11.3209457, 43.3808517, -53.4421158, 49.3804970
3: -17.3363628, 41.4867477, -19.8415718, 46.6844025, -64.0207596, 61.3283195
4: -16.3067322, 39.1788635, -18.4466991, 44.6605988, -60.9673271, 57.6255646

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B1_B1_A2_A1_A2_A1

### Relational analysis result of IS_B1_B1_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4244684, upper bound: 57.4244684
time: 0.58 seconds

## Relational analysis of IS_B1_B1_A2_A1_A2_A2

### Relational analysis result of IS_B1_B1_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4244684, upper bound: 57.4529045
time: 0.54 seconds

## BFS IS instance: IS_B1_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -9.5402002, 40.1481400, -8.4716768, 37.0284729, -46.5686722, 48.6198158
1: -12.1616077, 45.4533920, -10.8944139, 41.9450340, -54.1066360, 56.3478050
2: -11.9537477, 45.0598640, -10.6430874, 41.5497551, -53.5035019, 55.7029457
3: -20.5757256, 48.4900742, -18.8666878, 44.6714020, -65.2471237, 67.3567505
4: -19.1213646, 46.3223228, -17.3895168, 42.7868614, -61.9082222, 63.7118378

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B1_B1_A2_A2_B1_B1

### Relational analysis result of IS_B1_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5080668, upper bound: 57.4541081
time: 0.66 seconds

## Relational analysis of IS_B1_B1_A2_A2_B1_B2

### Relational analysis result of IS_B1_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5080670, upper bound: 57.5106913
time: 0.57 seconds

## BFS IS instance: IS_B1_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -9.6735115, 40.6550446, -7.9421101, 34.8733368, -44.5468483, 48.5971527
1: -12.3292837, 46.0220909, -10.1875982, 39.5196991, -51.8489799, 56.2096901
2: -12.1157894, 45.6436424, -10.0123949, 38.9890938, -51.1048813, 55.6560326
3: -20.8493938, 49.0880966, -17.6466942, 42.1907463, -63.0401382, 66.7347717
4: -19.3699036, 46.9194641, -16.4365597, 40.1360397, -59.5059433, 63.3560257

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_B1_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_B1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_B1_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B1_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B1_B1_A2_A2_B2_A1

### Relational analysis result of IS_B1_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5319804, upper bound: 57.5196555
time: 0.62 seconds

## Relational analysis of IS_B1_B1_A2_A2_B2_A2

### Relational analysis result of IS_B1_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5309281, upper bound: 57.5185887
time: 0.57 seconds

## BFS IS instance: IS_B1_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -9.8234758, 41.2086678, -9.7457952, 41.8517838, -51.6752510, 50.9544525
1: -12.5056238, 46.6563835, -12.5100403, 47.3378220, -59.8434372, 59.1664200
2: -12.2946529, 46.3034515, -12.1974192, 47.1026192, -59.3972702, 58.5008698
3: -21.2287254, 49.7448921, -21.4156284, 50.3785477, -71.6072693, 71.1605148
4: -19.8178635, 47.6446228, -19.7740021, 48.4451981, -68.2630615, 67.4186249

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B1_B2_A1_A2_B1_B1

### Relational analysis result of IS_B1_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3105501, upper bound: 57.0933261
time: 0.57 seconds

## Relational analysis of IS_B1_B2_A1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_B2_A1_A2_B1_A1

### Relational analysis result of IS_B1_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3270899, upper bound: 57.1283201
time: 0.56 seconds

## Relational analysis of IS_B1_B2_A1_A2_B1_A2

### Relational analysis result of IS_B1_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3498627, upper bound: 57.1368771
time: 0.60 seconds

## BFS IS instance: IS_B1_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -9.7674541, 40.9984703, -9.0554609, 38.8386421, -48.6060905, 50.0539246
1: -12.4357300, 46.4195595, -11.5887051, 43.9474831, -56.3832092, 58.0082626
2: -12.2277584, 46.0612373, -11.4078131, 43.5934486, -55.8212051, 57.4690514
3: -21.1118526, 49.4976578, -19.7838039, 46.9009323, -68.0127716, 69.2814636
4: -19.7159653, 47.3928604, -18.5310402, 44.8079567, -64.5239182, 65.9239044

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 43

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_B2_A1_A2_B2_A1

### Relational analysis result of IS_B1_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4244180, upper bound: 57.2938248
time: 0.57 seconds

## Relational analysis of IS_B1_B2_A1_A2_B2_A2

### Relational analysis result of IS_B1_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4343307, upper bound: 57.2982279
time: 0.77 seconds

## BFS IS instance: IS_B1_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -11.5261259, 47.8434143, -9.0554609, 38.8386421, -50.3647652, 56.8988686
1: -14.6432667, 54.1182861, -11.5887051, 43.9474831, -58.5907440, 65.7069931
2: -14.3591595, 53.9464798, -11.4078131, 43.5934486, -57.9526024, 65.3542938
3: -24.7230072, 57.5202217, -19.7838039, 46.9009323, -71.6239243, 77.3040237
4: -22.9618893, 55.5154381, -18.5310402, 44.8079567, -67.7698364, 74.0464783

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_B2_A2_A2_B2_A1

### Relational analysis result of IS_B1_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3042203, upper bound: 57.2493876
time: 0.66 seconds

## Relational analysis of IS_B1_B2_A2_A2_B2_A2

### Relational analysis result of IS_B1_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4018899, upper bound: 57.2870526
time: 0.63 seconds

## BFS IS instance: IS_B2_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -9.4956207, 40.8820877, -6.4694872, 28.4099197, -37.9055405, 47.3515739
1: -12.1049776, 46.2512856, -8.3112011, 32.2974014, -44.4023781, 54.5624847
2: -11.9203882, 45.8358002, -8.2349968, 31.6430893, -43.5634766, 54.0707970
3: -20.7166710, 49.2538376, -14.2844219, 34.6641693, -55.3808403, 63.5382614
4: -19.3075275, 47.0606422, -13.5163860, 32.6297150, -51.9372406, 60.5770264

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_B1_B1_B1_A1_A1

### Relational analysis result of IS_B2_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4781464, upper bound: 57.5070317
time: 0.57 seconds

## Relational analysis of IS_B2_B1_B1_B1_A1_A2

### Relational analysis result of IS_B2_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4781464, upper bound: 57.5070317
time: 0.59 seconds

## BFS IS instance: IS_B2_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -11.1461554, 46.0641098, -6.3032088, 27.8147678, -38.9609222, 52.3673172
1: -14.1850080, 52.1059799, -8.0949860, 31.6320152, -45.8170242, 60.2009659
2: -13.8932638, 51.9078407, -8.0360203, 30.9520950, -44.8453598, 59.9438591
3: -23.8666573, 55.4933815, -13.9376068, 33.9712944, -57.8379517, 69.4309692
4: -22.0540752, 53.3813171, -13.2225552, 31.9114590, -53.9655342, 66.6038742

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_B1_B1_B1_A2_A1

### Relational analysis result of IS_B2_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4709592, upper bound: 57.5040013
time: 1.08 seconds

## Relational analysis of IS_B2_B1_B1_B1_A2_A2

### Relational analysis result of IS_B2_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4709592, upper bound: 57.5040013
time: 0.60 seconds

## BFS IS instance: IS_B2_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -10.1770201, 42.4380188, -8.1467390, 35.0198555, -45.1968765, 50.5847549
1: -12.9492474, 48.0443954, -10.4393291, 39.6870842, -52.6363297, 58.4837265
2: -12.7204800, 47.7395439, -10.2659464, 39.2754898, -51.9959641, 58.0054855
3: -21.9290562, 51.2056084, -17.7922592, 42.2942581, -64.2233124, 68.9978638
4: -20.4464874, 49.1660233, -16.5044823, 40.4799461, -60.9264297, 65.6705017

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_B1_B1_B2_A1_A1

### Relational analysis result of IS_B2_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4781464, upper bound: 57.5161919
time: 0.53 seconds

## Relational analysis of IS_B2_B1_B1_B2_A1_A2

### Relational analysis result of IS_B2_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4709592, upper bound: 57.5131616
time: 0.58 seconds

## BFS IS instance: IS_B2_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -11.9793425, 49.4708633, -8.1467390, 35.0198555, -46.9991989, 57.6175957
1: -15.2106190, 55.9559326, -10.4393291, 39.6870842, -54.8977051, 66.3952408
2: -14.9106045, 55.8369446, -10.2659464, 39.2754898, -54.1860962, 66.1028900
3: -25.6317139, 59.4555740, -17.7922592, 42.2942581, -67.9259720, 77.2478333
4: -23.7928352, 57.5261345, -16.5044823, 40.4799461, -64.2727814, 74.0306015

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_B1_B1_B2_A2_A1

### Relational analysis result of IS_B2_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4781464, upper bound: 57.5070317
time: 0.61 seconds

## Relational analysis of IS_B2_B1_B1_B2_A2_A2

### Relational analysis result of IS_B2_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4709592, upper bound: 57.5040013
time: 0.56 seconds

## BFS IS instance: IS_B2_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -10.1770201, 42.4380188, -7.1907134, 31.1665955, -41.3436165, 49.6287308
1: -12.9492474, 48.0443954, -9.2092686, 35.3581696, -48.3074188, 57.2536621
2: -12.7204800, 47.7395439, -9.1051140, 34.7669830, -47.4874535, 56.8446579
3: -21.9290562, 51.2056084, -15.7794971, 37.8664360, -59.7954941, 66.9850922
4: -20.4464874, 49.1660233, -14.8050480, 35.8460159, -56.2925034, 63.9710464

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_B1_B2_B1_A1_A1

### Relational analysis result of IS_B2_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4783850, upper bound: 57.5070317
time: 0.58 seconds

## Relational analysis of IS_B2_B1_B2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B2_B1_B2_B1_A1_B1

### Relational analysis result of IS_B2_B1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4579380, upper bound: 57.4645076
time: 0.62 seconds

## Relational analysis of IS_B2_B1_B2_B1_A1_B2

### Relational analysis result of IS_B2_B1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4711978, upper bound: 57.5040013
time: 0.57 seconds

## BFS IS instance: IS_B2_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -11.9793425, 49.4708633, -7.1907134, 31.1665955, -43.1459389, 56.6615753
1: -15.2106190, 55.9559326, -9.2092686, 35.3581696, -50.5687866, 65.1651840
2: -14.9106045, 55.8369446, -9.1051140, 34.7669830, -49.6775894, 64.9420624
3: -25.6317139, 59.4555740, -15.7794971, 37.8664360, -63.4981499, 75.2350616
4: -23.7928352, 57.5261345, -14.8050480, 35.8460159, -59.6388512, 72.3311539

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_B1_B2_B1_A2_A1

### Relational analysis result of IS_B2_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4783850, upper bound: 57.5070317
time: 0.59 seconds

## Relational analysis of IS_B2_B1_B2_B1_A2_A2

### Relational analysis result of IS_B2_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4711978, upper bound: 57.5040013
time: 0.57 seconds

## BFS IS instance: IS_B2_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -10.1770201, 42.4380188, -8.9674406, 38.0639191, -48.2409363, 51.4054604
1: -12.9492474, 48.0443954, -11.4526777, 43.0664177, -56.0156631, 59.4970703
2: -12.7204800, 47.7395439, -11.2560825, 42.7363510, -55.4568253, 58.9956207
3: -21.9290562, 51.2056084, -19.4629898, 45.9011917, -67.8302383, 70.6685944
4: -20.4464874, 49.1660233, -18.0676479, 44.0224457, -64.4689255, 67.2336578

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_B1_B2_B2_A1_A1

### Relational analysis result of IS_B2_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4783850, upper bound: 57.5087039
time: 0.56 seconds

## Relational analysis of IS_B2_B1_B2_B2_A1_A2

### Relational analysis result of IS_B2_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4711978, upper bound: 57.5056735
time: 0.60 seconds

## BFS IS instance: IS_B2_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -11.9793425, 49.4708633, -8.9674406, 38.0639191, -50.0432625, 58.4383049
1: -15.2106190, 55.9559326, -11.4526777, 43.0664177, -58.2770386, 67.4085999
2: -14.9106045, 55.8369446, -11.2560825, 42.7363510, -57.6469574, 67.0930252
3: -25.6317139, 59.4555740, -19.4629898, 45.9011917, -71.5328903, 78.9185638
4: -23.7928352, 57.5261345, -18.0676479, 44.0224457, -67.8152771, 75.5937576

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_B1_B2_B2_A2_A1

### Relational analysis result of IS_B2_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4807533, upper bound: 57.4721431
time: 0.89 seconds

## Relational analysis of IS_B2_B1_B2_B2_A2_A2

### Relational analysis result of IS_B2_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4167163, upper bound: 57.4621409
time: 0.59 seconds

## BFS IS instance: IS_B2_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -7.9979382, 35.2653389, -11.8374672, 48.4148979, -56.4128342, 47.1027985
1: -10.2283983, 39.9275208, -15.0364037, 54.8208542, -65.0492401, 54.9639244
2: -10.1142387, 39.3997955, -14.8005733, 54.6966438, -64.8108749, 54.2003708
3: -17.6173477, 42.6080742, -25.2378845, 58.3669090, -75.9842529, 67.8459549
4: -16.4742813, 40.4700203, -23.5150375, 56.6646614, -73.1389313, 63.9850578

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_B2_B1_A1_A1_A1

### Relational analysis result of IS_B2_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4475262, upper bound: 57.4958876
time: 0.74 seconds

## Relational analysis of IS_B2_B2_B1_A1_A1_A2

### Relational analysis result of IS_B2_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4475262, upper bound: 57.5125861
time: 0.89 seconds

## BFS IS instance: IS_B2_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -11.8147278, 49.3174438, -11.8374672, 48.4148979, -60.2296257, 61.1548996
1: -15.0285769, 55.7443161, -15.0364037, 54.8208542, -69.8494186, 70.7807159
2: -14.7957783, 55.5128517, -14.8005733, 54.6966438, -69.4924240, 70.3134232
3: -25.4016705, 59.3138733, -25.2378845, 58.3669090, -83.7685699, 84.5517578
4: -23.5464191, 57.1426392, -23.5150375, 56.6646614, -80.2110748, 80.6576767

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_B2_B1_A1_A2_A1

### Relational analysis result of IS_B2_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4475262, upper bound: 57.4958876
time: 0.58 seconds

## Relational analysis of IS_B2_B2_B1_A1_A2_A2

### Relational analysis result of IS_B2_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4475262, upper bound: 57.5125861
time: 0.78 seconds

## BFS IS instance: IS_B2_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -8.2794256, 35.6002388, -11.5214596, 47.2093582, -55.4887772, 47.1216965
1: -10.5690498, 40.4112511, -14.6393099, 53.4600296, -64.0290680, 55.0505600
2: -10.4417515, 39.8581009, -14.4142208, 53.3048897, -63.7466431, 54.2723198
3: -17.9854412, 43.1620216, -24.5794067, 56.9279633, -74.9133987, 67.7414246
4: -16.8522739, 40.9721947, -22.9001389, 55.2124405, -72.0646973, 63.8723145

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_B2_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_B2_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_B2_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B2_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B2_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B2_B2_B1_A2_A1_B1

### Relational analysis result of IS_B2_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4958876, upper bound: 57.4475262
time: 0.56 seconds

## Relational analysis of IS_B2_B2_B1_A2_A1_B2

### Relational analysis result of IS_B2_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4958876, upper bound: 57.5209702
time: 0.62 seconds

## BFS IS instance: IS_B2_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -10.0152340, 41.9537582, -11.5214596, 47.2093582, -57.2245941, 53.4752121
1: -12.7629414, 47.4995117, -14.6393099, 53.4600296, -66.2229691, 62.1388206
2: -12.5277824, 47.1610222, -14.4142208, 53.3048897, -65.8326721, 61.5752220
3: -21.5519047, 50.6241608, -24.5794067, 56.9279633, -78.4798508, 75.2035675
4: -19.9791126, 48.4856224, -22.9001389, 55.2124405, -75.1915512, 71.3857574

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_B2_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_B2_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_B2_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B2_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B2_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B2_B2_B1_A2_A2_B1

### Relational analysis result of IS_B2_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4958876, upper bound: 57.4510926
time: 0.61 seconds

## Relational analysis of IS_B2_B2_B1_A2_A2_B2

### Relational analysis result of IS_B2_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4958876, upper bound: 57.5209702
time: 0.59 seconds

## BFS IS instance: IS_B2_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -11.0797024, 45.9357185, -9.4136696, 40.7493515, -51.8290520, 55.3493881
1: -14.1072273, 51.9529762, -12.0109186, 46.1312408, -60.2384682, 63.9638901
2: -13.8058453, 51.8173180, -11.8872576, 45.6063766, -59.4122238, 63.7045746
3: -23.7680473, 55.2296677, -20.5318451, 49.1757927, -72.9438400, 75.7615128
4: -21.9876976, 53.3685150, -19.2241611, 46.8254166, -68.8131104, 72.5926743

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_B2_B2_A1_B1_B1

### Relational analysis result of IS_B2_B2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5100452, upper bound: 57.4824912
time: 0.61 seconds

## Relational analysis of IS_B2_B2_B2_A1_B1_B2

### Relational analysis result of IS_B2_B2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5087039, upper bound: 57.4790815
time: 0.54 seconds

## BFS IS instance: IS_B2_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -10.8067226, 44.9140625, -10.6506176, 44.4792557, -55.2859764, 55.5646820
1: -13.7622919, 50.7994652, -13.6080704, 50.3465958, -64.1088867, 64.4075317
2: -13.4759121, 50.6409035, -13.3728914, 50.0002480, -63.4761581, 64.0137863
3: -23.2068501, 54.0196686, -22.9257927, 53.6933289, -76.9001770, 76.9454651
4: -21.4777794, 52.1379700, -21.2217865, 51.4474106, -72.9251862, 73.3597565

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_B2_B2_A1_B2_B1

### Relational analysis result of IS_B2_B2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5226226, upper bound: 57.5196516
time: 0.89 seconds

## Relational analysis of IS_B2_B2_B2_A1_B2_B2

### Relational analysis result of IS_B2_B2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5056735, upper bound: 57.4718943
time: 0.52 seconds

## BFS IS instance: IS_B2_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -12.2001457, 49.6175003, -12.4338989, 51.4096909, -63.6098366, 62.0513954
1: -15.4847717, 56.1783791, -15.7874660, 58.1604767, -73.6452484, 71.9658279
2: -15.2424402, 56.1024132, -15.5188274, 57.9909248, -73.2333679, 71.6212387
3: -25.9483662, 59.8135834, -26.5944901, 61.8139153, -87.7622833, 86.4080658
4: -24.1684036, 58.1983528, -24.7338448, 59.7358131, -83.9042206, 82.9321823

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_B2_B2_A2_A1_A1

### Relational analysis result of IS_B2_B2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4475262, upper bound: 57.5026646
time: 0.57 seconds

## Relational analysis of IS_B2_B2_B2_A2_A1_A2

### Relational analysis result of IS_B2_B2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5209701, upper bound: 57.5277472
time: 0.72 seconds

## BFS IS instance: IS_B2_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -12.8179855, 52.7224007, -12.4338989, 51.4096909, -64.2276764, 65.1562805
1: -16.2629013, 59.6412125, -15.7874660, 58.1604767, -74.4233780, 75.4286728
2: -15.9842758, 59.5239716, -15.5188274, 57.9909248, -73.9752045, 75.0427933
3: -27.3497543, 63.3827057, -26.5944901, 61.8139153, -89.1636658, 89.9771805
4: -25.4248123, 61.3837891, -24.7338448, 59.7358131, -85.1606293, 86.1176300

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_B2_B2_A2_A2_A1

### Relational analysis result of IS_B2_B2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4475262, upper bound: 57.5193631
time: 0.61 seconds

## Relational analysis of IS_B2_B2_B2_A2_A2_A2

### Relational analysis result of IS_B2_B2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5209701, upper bound: 57.5313136
time: 0.61 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.63 seconds
IS_B1_B1_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.4642506, upper bound: 57.4896601
IS_B1_B1_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.4657174, upper bound: 57.4969617
IS_B1_B1_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.4417602, upper bound: 57.4874880
IS_B1_B1_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.4657174, upper bound: 57.5051435
IS_B1_B1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.2644628, upper bound: 57.3971059
IS_B1_B1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.2659297, upper bound: 57.4044075
IS_B1_B1_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.2644628, upper bound: 57.4994701
IS_B1_B1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.2659297, upper bound: 57.5108207
IS_B1_B1_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.4897438, upper bound: 57.4483449
IS_B1_B1_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.4897438, upper bound: 57.4527688
IS_B1_B1_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.4244684, upper bound: 57.4244684
IS_B1_B1_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.4244684, upper bound: 57.4529045
IS_B1_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.5080668, upper bound: 57.4541081
IS_B1_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.5080670, upper bound: 57.5106913
IS_B1_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.5319804, upper bound: 57.5196555
IS_B1_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.5309281, upper bound: 57.5185887
IS_B1_B2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.3270899, upper bound: 57.1283201
IS_B1_B2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.3498627, upper bound: 57.1368771
IS_B1_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.4244180, upper bound: 57.2938248
IS_B1_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.4343307, upper bound: 57.2982279
IS_B1_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.3042203, upper bound: 57.2493876
IS_B1_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.4018899, upper bound: 57.2870526
IS_B2_B1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.4781464, upper bound: 57.5070317
IS_B2_B1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.4781464, upper bound: 57.5070317
IS_B2_B1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.4709592, upper bound: 57.5040013
IS_B2_B1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.4709592, upper bound: 57.5040013
IS_B2_B1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.4781464, upper bound: 57.5161919
IS_B2_B1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.4709592, upper bound: 57.5131616
IS_B2_B1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.4781464, upper bound: 57.5070317
IS_B2_B1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.4709592, upper bound: 57.5040013
IS_B2_B1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.4579380, upper bound: 57.4645076
IS_B2_B1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.4711978, upper bound: 57.5040013
IS_B2_B1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.4783850, upper bound: 57.5070317
IS_B2_B1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.4711978, upper bound: 57.5040013
IS_B2_B1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.4783850, upper bound: 57.5087039
IS_B2_B1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.4711978, upper bound: 57.5056735
IS_B2_B1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.4807533, upper bound: 57.4721431
IS_B2_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.4167163, upper bound: 57.4621409
IS_B2_B2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.4475262, upper bound: 57.4958876
IS_B2_B2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.4475262, upper bound: 57.5125861
IS_B2_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.4475262, upper bound: 57.4958876
IS_B2_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.4475262, upper bound: 57.5125861
IS_B2_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.4958876, upper bound: 57.4475262
IS_B2_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.4958876, upper bound: 57.5209702
IS_B2_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.4958876, upper bound: 57.4510926
IS_B2_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.4958876, upper bound: 57.5209702
IS_B2_B2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.5100452, upper bound: 57.4824912
IS_B2_B2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.5087039, upper bound: 57.4790815
IS_B2_B2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.5226226, upper bound: 57.5196516
IS_B2_B2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.5056735, upper bound: 57.4718943
IS_B2_B2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.4475262, upper bound: 57.5026646
IS_B2_B2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.5209701, upper bound: 57.5277472
IS_B2_B2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.4475262, upper bound: 57.5193631
IS_B2_B2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 0, lower bound: -57.5209701, upper bound: 57.5313136

## BFS IS instance: IS_B1_B1_A1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -5.9654880, 28.0278091, -8.6871595, 37.8292160, -43.7947044, 36.7149696
1: -7.6159410, 31.9026966, -11.1691742, 42.8474350, -50.4633713, 43.0718689
2: -7.6470280, 31.0369282, -10.9024353, 42.4748993, -50.1219254, 41.9393616
3: -13.5357962, 34.1880798, -19.3151951, 45.6187515, -59.1545486, 53.5032730
4: -12.9773741, 31.7998180, -17.7888889, 43.7372093, -56.7145844, 49.5886955

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_B1_B1_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B1_B1_A1_A1_A1_B1_B1

### Relational analysis result of IS_B1_B1_A1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4439833, upper bound: 57.4329935
time: 0.51 seconds

## Relational analysis of IS_B1_B1_A1_A1_A1_B1_B2

### Relational analysis result of IS_B1_B1_A1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4439833, upper bound: 57.4896601
time: 0.52 seconds

## BFS IS instance: IS_B1_B1_A1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -6.0675859, 28.3898735, -8.1397219, 35.6136017, -41.6811867, 36.5295944
1: -7.7485080, 32.3027954, -10.4412851, 40.3513832, -48.0998840, 42.7440796
2: -7.7708387, 31.4510880, -10.2504139, 39.8437119, -47.6145515, 41.7014999
3: -13.7540531, 34.6235008, -18.0642223, 43.0623169, -56.8163681, 52.6877136
4: -13.1703291, 32.2367897, -16.8020802, 41.0150299, -54.1853600, 49.0388603

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_B1_B1_A1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_B1_A1_A1_A1_B2_A1

### Relational analysis result of IS_B1_B1_A1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4497270, upper bound: 57.4919611
time: 0.62 seconds

## Relational analysis of IS_B1_B1_A1_A1_A1_B2_A2

### Relational analysis result of IS_B1_B1_A1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4497270, upper bound: 57.4969617
time: 0.67 seconds

## BFS IS instance: IS_B1_B1_A1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -5.9486384, 27.6570396, -9.1118717, 38.9983978, -44.9470367, 36.7689095
1: -7.5847530, 31.4490318, -11.6704311, 44.1492233, -51.7339783, 43.1194611
2: -7.6243067, 30.6088829, -11.4234848, 43.7770653, -51.4013710, 42.0323677
3: -13.3943548, 33.6805496, -20.0300770, 47.0839348, -60.4782677, 53.7106247
4: -12.7465086, 31.4047012, -18.6057186, 45.0663376, -57.8128433, 50.0104179

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B1_B1_A1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_B1_A1_A1_A2_A1_B1

### Relational analysis result of IS_B1_B1_A1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4402933, upper bound: 57.4801864
time: 0.62 seconds

## Relational analysis of IS_B1_B1_A1_A1_A2_A1_B2

### Relational analysis result of IS_B1_B1_A1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4402933, upper bound: 57.4874880
time: 0.63 seconds

## BFS IS instance: IS_B1_B1_A1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -5.5963645, 26.3081665, -9.2238035, 39.4067192, -45.0030785, 35.5319710
1: -7.1001191, 29.9992542, -11.8112040, 44.6088829, -51.7090034, 41.8104591
2: -7.2258878, 29.0324535, -11.5591335, 44.2477875, -51.4736748, 40.5915871
3: -12.5410995, 32.1875839, -20.2589512, 47.5700760, -60.1111755, 52.4465294
4: -12.1948471, 29.6890259, -18.8161907, 45.5512772, -57.7461243, 48.5052185

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B1_B1_A1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_B1_A1_A1_A2_A2_B1

### Relational analysis result of IS_B1_B1_A1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4877973, upper bound: 57.4978419
time: 0.82 seconds

## Relational analysis of IS_B1_B1_A1_A1_A2_A2_B2

### Relational analysis result of IS_B1_B1_A1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4877973, upper bound: 57.5051435
time: 0.60 seconds

## BFS IS instance: IS_B1_B1_A1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -7.8669782, 35.4995689, -8.6871595, 37.8292160, -45.6961937, 44.1867256
1: -10.0885792, 40.1932335, -11.1691742, 42.8474350, -52.9360123, 51.3624077
2: -9.9576416, 39.6621132, -10.9024353, 42.4748993, -52.4325409, 50.5645485
3: -17.6324806, 42.9004478, -19.3151951, 45.6187515, -63.2512321, 62.2156448
4: -16.5203495, 40.7158279, -17.7888889, 43.7372093, -60.2575607, 58.5046997

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B1_B1_A1_A2_A1_B1_B1

### Relational analysis result of IS_B1_B1_A1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2441955, upper bound: 57.3404393
time: 0.55 seconds

## Relational analysis of IS_B1_B1_A1_A2_A1_B1_B2

### Relational analysis result of IS_B1_B1_A1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2441955, upper bound: 57.3971059
time: 0.55 seconds

## BFS IS instance: IS_B1_B1_A1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -7.9782028, 35.9070625, -8.1397219, 35.6136017, -43.5918045, 44.0467834
1: -10.2312508, 40.6491737, -10.4412851, 40.3513832, -50.5826340, 51.0904579
2: -10.0926924, 40.1304741, -10.2504139, 39.8437119, -49.9364052, 50.3808823
3: -17.8641090, 43.3818817, -18.0642223, 43.0623169, -60.9264259, 61.4460945
4: -16.7275486, 41.1993408, -16.8020802, 41.0150299, -57.7425728, 58.0014153

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_B1_B1_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_B1_B1_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B1_B1_A1_A2_A1_B2_B1

### Relational analysis result of IS_B1_B1_A1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2491961, upper bound: 57.3564297
time: 1.25 seconds

## Relational analysis of IS_B1_B1_A1_A2_A1_B2_B2

### Relational analysis result of IS_B1_B1_A1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2491961, upper bound: 57.4044075
time: 0.58 seconds

## BFS IS instance: IS_B1_B1_A1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -8.3375483, 36.6368942, -8.6871595, 37.8292160, -46.1667595, 45.3240547
1: -10.6374722, 41.4734917, -11.1691742, 42.8474350, -53.4849052, 52.6426659
2: -10.5258341, 40.9356232, -10.9024353, 42.4748993, -53.0007286, 51.8380585
3: -18.3378468, 44.2414742, -19.3151951, 45.6187515, -63.9565964, 63.5566559
4: -17.1834927, 42.0157013, -17.7888889, 43.7372093, -60.9207001, 59.8045883

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B1_B1_A1_A2_A2_B1_B1

### Relational analysis result of IS_B1_B1_A1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4731226, upper bound: 57.4423322
time: 0.57 seconds

## Relational analysis of IS_B1_B1_A1_A2_A2_B1_B2

### Relational analysis result of IS_B1_B1_A1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4731226, upper bound: 57.4994701
time: 0.59 seconds

## BFS IS instance: IS_B1_B1_A1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -8.4552927, 37.0844460, -8.1397219, 35.6136017, -44.0688934, 45.2241669
1: -10.7860126, 41.9743881, -10.4412851, 40.3513832, -51.1373978, 52.4156723
2: -10.6693573, 41.4484825, -10.2504139, 39.8437119, -50.5130653, 51.6988983
3: -18.5805569, 44.7684402, -18.0642223, 43.0623169, -61.6428719, 62.8326645
4: -17.4033432, 42.5424194, -16.8020802, 41.0150299, -58.4183502, 59.3444977

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_B1_B1_A1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B1_B1_A1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B1_B1_A1_A2_A2_B2_B1

### Relational analysis result of IS_B1_B1_A1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4877502, upper bound: 57.4628637
time: 0.60 seconds

## Relational analysis of IS_B1_B1_A1_A2_A2_B2_B2

### Relational analysis result of IS_B1_B1_A1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4877502, upper bound: 57.5108207
time: 0.57 seconds

## BFS IS instance: IS_B1_B1_A2_A1_A1_A1

### Backsubstitution after applying IS history:
0: -5.6862741, 26.5720062, -8.8738098, 38.0590744, -43.7453499, 35.4458122
1: -7.2837358, 30.3159943, -11.3669062, 43.0893440, -50.3730736, 41.6828957
2: -7.3231826, 29.3796921, -11.1374245, 42.6968994, -50.0200806, 40.5171165
3: -12.7803907, 32.5764503, -19.5241280, 45.9783096, -58.7587013, 52.1005783
4: -12.4527779, 30.0693321, -18.1578388, 43.9637909, -56.4165688, 48.2271614

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_B1_B1_A2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_B1_A2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_B1_B1_A2_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B1_B1_A2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B1_B1_A2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_B1_B1_A2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_B1_A2_A1_A1_A1_A1

### Relational analysis result of IS_B1_B1_A2_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4761469, upper bound: 57.4430971
time: 0.55 seconds

## Relational analysis of IS_B1_B1_A2_A1_A1_A1_A2

### Relational analysis result of IS_B1_B1_A2_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4897438, upper bound: 57.4483449
time: 0.79 seconds

## BFS IS instance: IS_B1_B1_A2_A1_A1_A2

### Backsubstitution after applying IS history:
0: -5.7091141, 26.1507530, -8.8738098, 38.0590744, -43.7681847, 35.0245590
1: -7.2901106, 29.8523808, -11.3669062, 43.0893440, -50.3794556, 41.2192764
2: -7.3597307, 28.8781662, -11.1374245, 42.6968994, -50.0566216, 40.0155830
3: -12.6524143, 32.1650085, -19.5241280, 45.9783096, -58.6307144, 51.6891365
4: -12.4052563, 29.5441093, -18.1578388, 43.9637909, -56.3690491, 47.7019463

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_B1_B1_A2_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_B1_B1_A2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_B1_A2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B1_B1_A2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_B1_B1_A2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_B1_A2_A1_A1_A2_A1

### Relational analysis result of IS_B1_B1_A2_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4761469, upper bound: 57.4430971
time: 0.56 seconds

## Relational analysis of IS_B1_B1_A2_A1_A1_A2_A2

### Relational analysis result of IS_B1_B1_A2_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4897438, upper bound: 57.4527688
time: 0.54 seconds

## BFS IS instance: IS_B1_B1_A2_A1_A2_A1

### Backsubstitution after applying IS history:
0: -7.6163511, 33.4761925, -9.0251799, 38.6557541, -46.2721024, 42.5013733
1: -9.7761698, 37.9952393, -11.5584135, 43.7627220, -53.5388908, 49.5536537
2: -9.6483345, 37.3633232, -11.3209457, 43.3808517, -53.0291748, 48.6842690
3: -16.8829327, 40.7489586, -19.8415718, 46.6844025, -63.5673370, 60.5905266
4: -15.8857355, 38.4501076, -18.4466991, 44.6605988, -60.5463295, 56.8968048

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_B1_B1_A2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_B1_B1_A2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_B1_A2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_B1_B1_A2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B1_B1_A2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B1_B1_A2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B1_B1_A2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B1_B1_A2_A1_A2_A1_B1

### Relational analysis result of IS_B1_B1_A2_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4244684, upper bound: 57.4244684
time: 0.57 seconds

## Relational analysis of IS_B1_B1_A2_A1_A2_A1_B2

### Relational analysis result of IS_B1_B1_A2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4244684, upper bound: 57.4244684
time: 0.56 seconds

## BFS IS instance: IS_B1_B1_A2_A1_A2_A2

### Backsubstitution after applying IS history:
0: -7.8097162, 33.6637039, -9.0251799, 38.6557541, -46.4654617, 42.6888847
1: -9.9945507, 38.2244110, -11.5584135, 43.7627220, -53.7572708, 49.7828255
2: -9.8847504, 37.5782623, -11.3209457, 43.3808517, -53.2656021, 48.8992081
3: -17.0487785, 40.9957657, -19.8415718, 46.6844025, -63.7331772, 60.8373260
4: -16.0561256, 38.6579552, -18.4466991, 44.6605988, -60.7167168, 57.1046524

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_B1_B1_A2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B1_B1_A2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_B1_B1_A2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_B1_A2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_B1_B1_A2_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B1_B1_A2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B1_B1_A2_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B1_B1_A2_A1_A2_A2_B1

### Relational analysis result of IS_B1_B1_A2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4244684, upper bound: 57.4529045
time: 0.61 seconds

## Relational analysis of IS_B1_B1_A2_A1_A2_A2_B2

### Relational analysis result of IS_B1_B1_A2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4244684, upper bound: 57.4529045
time: 0.58 seconds

## BFS IS instance: IS_B1_B1_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -9.5402002, 40.1481400, -5.6884518, 27.4653168, -37.0055161, 45.8365860
1: -12.1616077, 45.4533920, -7.2809191, 31.2659359, -43.4275398, 52.7343102
2: -11.9537477, 45.0598640, -7.3165393, 30.3996086, -42.3533554, 52.3764038
3: -20.5757256, 48.4900742, -13.0981560, 33.4030228, -53.9787445, 61.5882301
4: -19.1213646, 46.3223228, -12.4632511, 31.1486492, -50.2700119, 58.7855759

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B1_B1_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_B1_A2_A2_B1_B1_A1

### Relational analysis result of IS_B1_B1_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4455365, upper bound: 57.4207906
time: 0.57 seconds

## Relational analysis of IS_B1_B1_A2_A2_B1_B1_A2

### Relational analysis result of IS_B1_B1_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4455365, upper bound: 57.4541081
time: 0.59 seconds

## BFS IS instance: IS_B1_B1_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -9.5402002, 40.1481400, -6.6024351, 30.2522888, -39.7924881, 46.7505722
1: -12.1616077, 45.4533920, -8.4603643, 34.3885117, -46.5501175, 53.9137573
2: -11.9537477, 45.0598640, -8.4155941, 33.6497993, -45.6035461, 53.4754486
3: -20.5757256, 48.4900742, -14.8398304, 36.7894897, -57.3652115, 63.3299026
4: -19.1213646, 46.3223228, -13.9825315, 34.5861893, -53.7075539, 60.3048553

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_B1_A2_A2_B1_B2_A1

### Relational analysis result of IS_B1_B1_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4455366, upper bound: 57.4777092
time: 0.69 seconds

## Relational analysis of IS_B1_B1_A2_A2_B1_B2_A2

### Relational analysis result of IS_B1_B1_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4455366, upper bound: 57.5106913
time: 0.58 seconds

## BFS IS instance: IS_B1_B1_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -9.0417519, 37.5709839, -7.8601747, 34.5904846, -43.6322327, 45.4311600
1: -11.4478045, 42.6115150, -10.0842113, 39.2036171, -50.6514206, 52.6957245
2: -11.3106394, 42.1282501, -9.9135866, 38.6607857, -49.9714241, 52.0418320
3: -19.2548580, 45.6073799, -17.4812393, 41.8563271, -61.1111832, 63.0886192
4: -18.2657948, 43.1796036, -16.2881260, 39.7961464, -58.0619316, 59.4677200

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_B1_B1_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_B1_B1_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B1_B1_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B1_B1_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B1_B1_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B1_B1_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_B1_B1_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B1_B1_A2_A2_B2_A1_B1

### Relational analysis result of IS_B1_B1_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5319463, upper bound: 57.5196148
time: 0.59 seconds

## Relational analysis of IS_B1_B1_A2_A2_B2_A1_B2

### Relational analysis result of IS_B1_B1_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5077313, upper bound: 57.4539965
time: 0.58 seconds

## BFS IS instance: IS_B1_B1_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.4147587, 39.6639862, -7.9421101, 34.8733368, -44.2880936, 47.6060944
1: -11.9998913, 44.9106369, -10.1875982, 39.5196991, -51.5195923, 55.0982361
2: -11.8009129, 44.5011711, -10.0123949, 38.9890938, -50.7900085, 54.5135651
3: -20.3079472, 47.9158974, -17.6466942, 42.1907463, -62.4986954, 65.5625916
4: -18.8814545, 45.7507896, -16.4365597, 40.1360397, -59.0174942, 62.1873474

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B1_B1_A2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_B1_B1_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_B1_B1_A2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B1_B1_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_B1_B1_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B1_B1_A2_A2_B2_A2_A1

### Relational analysis result of IS_B1_B1_A2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5033331, upper bound: 57.4526942
time: 0.52 seconds

## Relational analysis of IS_B1_B1_A2_A2_B2_A2_A2

### Relational analysis result of IS_B1_B1_A2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5058915, upper bound: 57.4532721
time: 0.61 seconds

## BFS IS instance: IS_B1_B2_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -8.8402529, 37.7630539, -9.6252708, 41.4112663, -50.2515182, 47.3883209
1: -11.2974596, 42.7681274, -12.3575668, 46.8403702, -58.1378212, 55.1256943
2: -11.0862007, 42.3602257, -12.0518417, 46.5954857, -57.6816864, 54.4120636
3: -19.3324356, 45.5596771, -21.1698074, 49.8535576, -69.1859894, 66.7294769
4: -17.8893318, 43.6359253, -19.5500412, 47.9230080, -65.8123398, 63.1859589

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B1_B2_A1_A2_B1_A1_B1

### Relational analysis result of IS_B1_B2_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2921143, upper bound: 57.0854927
time: 0.59 seconds

## Relational analysis of IS_B1_B2_A1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B1_B2_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B1_B2_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B1_B2_A1_A2_B1_A1_A1

### Relational analysis result of IS_B1_B2_A1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3269722, upper bound: 57.1283201
time: 0.62 seconds

## Relational analysis of IS_B1_B2_A1_A2_B1_A1_A2

### Relational analysis result of IS_B1_B2_A1_A2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -57.2463794, upper bound: 57.0958706
time: 0.56 seconds

## BFS IS instance: IS_B1_B2_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -8.4693804, 36.1834831, -9.7457952, 41.8517838, -50.3211555, 45.9292717
1: -10.8127460, 41.0070839, -12.5100403, 47.3378220, -58.1505661, 53.5171242
2: -10.6570034, 40.5084305, -12.1974192, 47.1026192, -57.7596169, 52.7058487
3: -18.4758568, 43.8035660, -21.4156284, 50.3785477, -68.8544006, 65.2191849
4: -17.2884102, 41.7039909, -19.7740021, 48.4451981, -65.7336121, 61.4779930

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B1_B2_A1_A2_B1_A2_B1

### Relational analysis result of IS_B1_B2_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3105501, upper bound: 57.0933261
time: 0.56 seconds

## Relational analysis of IS_B1_B2_A1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B1_B2_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B1_B2_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B1_B2_A1_A2_B1_A2_A1

### Relational analysis result of IS_B1_B2_A1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3498627, upper bound: 57.1364882
time: 0.69 seconds

## Relational analysis of IS_B1_B2_A1_A2_B1_A2_A2

### Relational analysis result of IS_B1_B2_A1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3389090, upper bound: 57.1327151
time: 0.61 seconds

## BFS IS instance: IS_B1_B2_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -8.5403042, 36.1297760, -9.0554609, 38.8386421, -47.3789444, 45.1852303
1: -10.9181147, 40.9794960, -11.5887051, 43.9474831, -54.8655968, 52.5681992
2: -10.7451744, 40.5320091, -11.4078131, 43.5934486, -54.3386230, 51.9398232
3: -18.5557365, 43.7732811, -19.7838039, 46.9009323, -65.4566422, 63.5570602
4: -17.3938255, 41.7675552, -18.5310402, 44.8079567, -62.2017670, 60.2985954

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B1_B2_A1_A2_B2_A1_B1

### Relational analysis result of IS_B1_B2_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3996015, upper bound: 57.2564692
time: 0.61 seconds

## Relational analysis of IS_B1_B2_A1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B1_B2_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_B2_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_B2_A1_A2_B2_A1_B1

### Relational analysis result of IS_B1_B2_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3433837, upper bound: 57.1442029
time: 0.66 seconds

## Relational analysis of IS_B1_B2_A1_A2_B2_A1_B2

### Relational analysis result of IS_B1_B2_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4244180, upper bound: 57.2938248
time: 0.53 seconds

## BFS IS instance: IS_B1_B2_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.0252848, 38.0982437, -9.0554609, 38.8386421, -47.8639259, 47.1536980
1: -11.5037489, 43.1570396, -11.5887051, 43.9474831, -55.4512329, 54.7457428
2: -11.3232031, 42.7301025, -11.4078131, 43.5934486, -54.9166527, 54.1379166
3: -19.5718575, 46.0672531, -19.7838039, 46.9009323, -66.4727707, 65.8510513
4: -18.2861252, 44.0010490, -18.5310402, 44.8079567, -63.0940781, 62.5320892

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 43

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B1_B2_A1_A2_B2_A2_B1

### Relational analysis result of IS_B1_B2_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4035307, upper bound: 57.2595995
time: 0.57 seconds

## Relational analysis of IS_B1_B2_A1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B1_B2_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_B2_A1_A2_B2_A2_B1

### Relational analysis result of IS_B1_B2_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4154320, upper bound: 57.2504996
time: 0.60 seconds

## Relational analysis of IS_B1_B2_A1_A2_B2_A2_B2

### Relational analysis result of IS_B1_B2_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4154320, upper bound: 57.2982279
time: 0.60 seconds

## BFS IS instance: IS_B1_B2_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -10.2012281, 42.5499268, -9.0554609, 38.8386421, -49.0398712, 51.6053848
1: -12.9891100, 48.2006073, -11.5887051, 43.9474831, -56.9365921, 59.7893143
2: -12.7566290, 47.9385490, -11.4078131, 43.5934486, -56.3500786, 59.3463631
3: -21.9618587, 51.3005753, -19.7838039, 46.9009323, -68.8627777, 71.0843811
4: -20.4330769, 49.4441872, -18.5310402, 44.8079567, -65.2410355, 67.9752197

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B1_B2_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B1_B2_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B1_B2_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B1_B2_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_B2_A2_A2_B2_A1_B1

### Relational analysis result of IS_B1_B2_A2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -57.2231860, upper bound: 57.0997657
time: 0.63 seconds

## Relational analysis of IS_B1_B2_A2_A2_B2_A1_B2

### Relational analysis result of IS_B1_B2_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3042203, upper bound: 57.2493876
time: 0.66 seconds

## BFS IS instance: IS_B1_B2_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -10.7570248, 44.8572044, -9.0554609, 38.8386421, -49.5956612, 53.9126625
1: -13.6746721, 50.7558975, -11.5887051, 43.9474831, -57.6221504, 62.3446045
2: -13.4192657, 50.5207176, -11.4078131, 43.5934486, -57.0127106, 61.9285316
3: -23.1309681, 53.9832993, -19.7838039, 46.9009323, -70.0318909, 73.7670975
4: -21.4761810, 51.9948349, -18.5310402, 44.8079567, -66.2841339, 70.5258560

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_B1_B2_A2_A2_B2_A2_B1

### Relational analysis result of IS_B1_B2_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3557734, upper bound: 57.2426504
time: 0.56 seconds

## Relational analysis of IS_B1_B2_A2_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B1_B2_A2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_B2_A2_A2_B2_A2_B1

### Relational analysis result of IS_B1_B2_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3829913, upper bound: 57.2393244
time: 0.60 seconds

## Relational analysis of IS_B1_B2_A2_A2_B2_A2_B2

### Relational analysis result of IS_B1_B2_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3829913, upper bound: 57.2870526
time: 0.62 seconds

## BFS IS instance: IS_B2_B1_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -7.1131306, 31.6738453, -6.4694872, 28.4099197, -35.5230484, 38.1433334
1: -9.0788956, 35.9604263, -8.3112011, 32.2974014, -41.3762970, 44.2716293
2: -9.0415535, 35.2259827, -8.2349968, 31.6430893, -40.6846390, 43.4609795
3: -15.7540169, 38.5447044, -14.2844219, 34.6641693, -50.4181786, 52.8291245
4: -14.9598646, 36.1675186, -13.5163860, 32.6297150, -47.5895729, 49.6839066

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_B1_B1_B1_A1_A1_A1

### Relational analysis result of IS_B2_B1_B1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4044752, upper bound: 57.4714887
time: 0.91 seconds

## Relational analysis of IS_B2_B1_B1_B1_A1_A1_A2

### Relational analysis result of IS_B2_B1_B1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4781464, upper bound: 57.5070317
time: 0.58 seconds

## BFS IS instance: IS_B2_B1_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -8.8355389, 38.4204979, -6.4694872, 28.4099197, -37.2454605, 44.8899841
1: -11.2725945, 43.4727173, -8.3112011, 32.2974014, -43.5699959, 51.7839203
2: -11.1283092, 43.0069008, -8.2349968, 31.6430893, -42.7714005, 51.2418976
3: -19.3558121, 46.3429070, -14.2844219, 34.6641693, -54.0199661, 60.6273270
4: -18.0807438, 44.1607170, -13.5163860, 32.6297150, -50.7104568, 57.6771011

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_B1_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_B1_B1_B1_A1_A2_A1

### Relational analysis result of IS_B2_B1_B1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4044752, upper bound: 57.4714887
time: 0.57 seconds

## Relational analysis of IS_B2_B1_B1_B1_A1_A2_A2

### Relational analysis result of IS_B2_B1_B1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4781464, upper bound: 57.5070317
time: 0.55 seconds

## BFS IS instance: IS_B2_B1_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -8.4568176, 36.0478821, -6.3032088, 27.8147678, -36.2715836, 42.3510895
1: -10.8113356, 40.8942490, -8.0949860, 31.6320152, -42.4433479, 48.9892349
2: -10.6619377, 40.3148651, -8.0360203, 30.9520950, -41.6140327, 48.3508835
3: -18.3744678, 43.7796822, -13.9376068, 33.9712944, -52.3457642, 57.7172890
4: -17.2096577, 41.4841347, -13.2225552, 31.9114590, -49.1211090, 54.7066879

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_B1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_B1_B1_B1_A2_A1_A1

### Relational analysis result of IS_B2_B1_B1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4299904, upper bound: 57.4781235
time: 0.65 seconds

## Relational analysis of IS_B2_B1_B1_B1_A2_A1_A2

### Relational analysis result of IS_B2_B1_B1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4709592, upper bound: 57.5040013
time: 0.60 seconds

## BFS IS instance: IS_B2_B1_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -10.5398588, 43.7455978, -6.3032088, 27.8147678, -38.3546257, 50.0488014
1: -13.4253931, 49.4864922, -8.0949860, 31.6320152, -45.0574074, 57.5814781
2: -13.1599941, 49.2285156, -8.0360203, 30.9520950, -44.1120911, 57.2645340
3: -22.6216278, 52.7429161, -13.9376068, 33.9712944, -56.5929222, 66.6805191
4: -20.9137344, 50.6451569, -13.2225552, 31.9114590, -52.8251953, 63.8677139

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_B1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_B2_B1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_B1_B1_B1_A2_A2_A1

### Relational analysis result of IS_B2_B1_B1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4299904, upper bound: 57.4781235
time: 0.59 seconds

## Relational analysis of IS_B2_B1_B1_B1_A2_A2_A2

### Relational analysis result of IS_B2_B1_B1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4709592, upper bound: 57.5040013
time: 0.83 seconds

## BFS IS instance: IS_B2_B1_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -7.1131306, 31.6738453, -8.1467390, 35.0198555, -42.1329880, 39.8205833
1: -9.0788956, 35.9604263, -10.4393291, 39.6870842, -48.7659798, 46.3997574
2: -9.0415535, 35.2259827, -10.2659464, 39.2754898, -48.3170395, 45.4919281
3: -15.7540169, 38.5447044, -17.7922592, 42.2942581, -58.0482750, 56.3369637
4: -14.9598646, 36.1675186, -16.5044823, 40.4799461, -55.4398079, 52.6720009

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_B1_B1_B2_A1_A1_B1

### Relational analysis result of IS_B2_B1_B1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4147628, upper bound: 57.4878504
time: 0.54 seconds

## Relational analysis of IS_B2_B1_B1_B2_A1_A1_B2

### Relational analysis result of IS_B2_B1_B1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4139802, upper bound: 57.4889256
time: 0.58 seconds

## BFS IS instance: IS_B2_B1_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -8.4568176, 36.0478821, -7.8793817, 34.0400429, -42.4968605, 43.9272652
1: -10.8113356, 40.8942490, -10.0996008, 38.5869942, -49.3983307, 50.9938469
2: -10.6619377, 40.3148651, -9.9428596, 38.1413078, -48.8032455, 50.2577171
3: -18.3744678, 43.7796822, -17.2384415, 41.1399307, -59.5143967, 61.0181236
4: -17.2096577, 41.4841347, -16.0126858, 39.3070259, -56.5166740, 57.4968185

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_B1_B1_B2_A1_A2_A1

### Relational analysis result of IS_B2_B1_B1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5011161, upper bound: 57.5192858
time: 0.54 seconds

## Relational analysis of IS_B2_B1_B1_B2_A1_A2_A2

### Relational analysis result of IS_B2_B1_B1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5216813, upper bound: 57.5301108
time: 0.58 seconds

## BFS IS instance: IS_B2_B1_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -8.8355389, 38.4204979, -8.1467390, 35.0198555, -43.8553925, 46.5672302
1: -11.2725945, 43.4727173, -10.4393291, 39.6870842, -50.9596786, 53.9120483
2: -11.1283092, 43.0069008, -10.2659464, 39.2754898, -50.4038010, 53.2728348
3: -19.3558121, 46.3429070, -17.7922592, 42.2942581, -61.6500702, 64.1351624
4: -18.0807438, 44.1607170, -16.5044823, 40.4799461, -58.5606880, 60.6651993

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_B1_B1_B2_A2_A1_B1

### Relational analysis result of IS_B2_B1_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4091528, upper bound: 57.4816168
time: 0.61 seconds

## Relational analysis of IS_B2_B1_B1_B2_A2_A1_B2

### Relational analysis result of IS_B2_B1_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4001957, upper bound: 57.4689572
time: 0.63 seconds

## BFS IS instance: IS_B2_B1_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -9.8936510, 41.4662666, -7.8793817, 34.0400429, -43.9336929, 49.3456459
1: -12.6075249, 46.9325066, -10.0996008, 38.5869942, -51.1945190, 57.0321083
2: -12.3815975, 46.5865746, -9.9428596, 38.1413078, -50.5229034, 56.5294304
3: -21.2981529, 50.0432014, -17.2384415, 41.1399307, -62.4380836, 67.2816467
4: -19.7658882, 47.8902588, -16.0126858, 39.3070259, -59.0729141, 63.9029388

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_B1_B1_B2_A2_A2_A1

### Relational analysis result of IS_B2_B1_B1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4299904, upper bound: 57.4781235
time: 0.63 seconds

## Relational analysis of IS_B2_B1_B1_B2_A2_A2_A2

### Relational analysis result of IS_B2_B1_B1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4709592, upper bound: 57.5040013
time: 0.65 seconds

## BFS IS instance: IS_B2_B1_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -10.1770201, 42.4380188, -4.6906605, 22.5372143, -32.7142334, 47.1286774
1: -12.9492474, 48.0443954, -5.9458561, 25.7812634, -38.7305107, 53.9902496
2: -12.7204800, 47.7395439, -6.1137819, 24.7862644, -37.5067368, 53.8533211
3: -21.9290562, 51.2056084, -10.5180454, 27.7272148, -49.6562729, 61.7236557
4: -20.4464874, 49.1660233, -10.3832111, 25.3039093, -45.7503815, 59.5492210

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_B1_B2_B1_A1_B1_A1

### Relational analysis result of IS_B2_B1_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4870197, upper bound: 57.4734334
time: 0.61 seconds

## Relational analysis of IS_B2_B1_B2_B1_A1_B1_A2

### Relational analysis result of IS_B2_B1_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5056954, upper bound: 57.4814567
time: 0.60 seconds

## BFS IS instance: IS_B2_B1_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -9.9040661, 41.3869324, -6.0265141, 27.1719227, -37.0759735, 47.4134331
1: -12.6045265, 46.8566895, -7.7019172, 30.9572868, -43.5618134, 54.5586052
2: -12.3889084, 46.5329971, -7.7265854, 30.0770111, -42.4659157, 54.2595787
3: -21.3612614, 49.9555511, -13.2806883, 33.3188248, -54.6800842, 63.2362366
4: -19.9208145, 47.9058762, -12.7388630, 30.9022064, -50.8230209, 60.6447372

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_B1_B2_B1_A1_B2_A1

### Relational analysis result of IS_B2_B1_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5002794, upper bound: 57.5129271
time: 0.53 seconds

## Relational analysis of IS_B2_B1_B2_B1_A1_B2_A2

### Relational analysis result of IS_B2_B1_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5189551, upper bound: 57.5209504
time: 0.60 seconds

## BFS IS instance: IS_B2_B1_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -8.8355389, 38.4204979, -7.1907134, 31.1665955, -40.0021362, 45.6112099
1: -11.2725945, 43.4727173, -9.2092686, 35.3581696, -46.6307640, 52.6819839
2: -11.1283092, 43.0069008, -9.1051140, 34.7669830, -45.8952942, 52.1120110
3: -19.3558121, 46.3429070, -15.7794971, 37.8664360, -57.2222404, 62.1224060
4: -18.0807438, 44.1607170, -14.8050480, 35.8460159, -53.9267578, 58.9657669

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_B1_B2_B1_A2_A1_A1

### Relational analysis result of IS_B2_B1_B2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4546673, upper bound: 57.4613972
time: 0.60 seconds

## Relational analysis of IS_B2_B1_B2_B1_A2_A1_A2

### Relational analysis result of IS_B2_B1_B2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3887422, upper bound: 57.4406043
time: 0.62 seconds

## BFS IS instance: IS_B2_B1_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -9.8936510, 41.4662666, -6.9861779, 30.3978500, -40.2915001, 48.4524460
1: -12.6075249, 46.9325066, -8.9459486, 34.4972801, -47.1048050, 55.8784561
2: -12.3815975, 46.5865746, -8.8603287, 33.8817940, -46.2633820, 55.4469032
3: -21.2981529, 50.0432014, -15.3459682, 36.9700928, -58.2682381, 65.3891678
4: -19.7658882, 47.8902588, -14.4331579, 34.9347687, -54.7006569, 62.3234177

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_B1_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_B1_B2_B1_A2_A2_B1

### Relational analysis result of IS_B2_B1_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4523134, upper bound: 57.4660028
time: 0.59 seconds

## Relational analysis of IS_B2_B1_B2_B1_A2_A2_B2

### Relational analysis result of IS_B2_B1_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4709592, upper bound: 57.5040013
time: 0.60 seconds

## BFS IS instance: IS_B2_B1_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -7.1131306, 31.6738453, -8.9674406, 38.0639191, -45.1770477, 40.6412849
1: -9.0788956, 35.9604263, -11.4526777, 43.0664177, -52.1453133, 47.4131012
2: -9.0415535, 35.2259827, -11.2560825, 42.7363510, -51.7779007, 46.4820633
3: -15.7540169, 38.5447044, -19.4629898, 45.9011917, -61.6552086, 58.0076942
4: -14.9598646, 36.1675186, -18.0676479, 44.0224457, -58.9822998, 54.2351646

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_B1_B2_B2_A1_A1_A1

### Relational analysis result of IS_B2_B1_B2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4383907, upper bound: 57.4930176
time: 0.62 seconds

## Relational analysis of IS_B2_B1_B2_B2_A1_A1_A2

### Relational analysis result of IS_B2_B1_B2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4383907, upper bound: 57.5100452
time: 0.60 seconds

## BFS IS instance: IS_B2_B1_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -8.4568176, 36.0478821, -8.7125311, 37.1107979, -45.5676155, 44.7604103
1: -10.8113356, 40.8942490, -11.1296597, 41.9948921, -52.8062248, 52.0239067
2: -10.6619377, 40.3148651, -10.9500465, 41.6366310, -52.2985649, 51.2649078
3: -18.3744678, 43.7796822, -18.9335423, 44.7786446, -63.1531143, 62.7132225
4: -17.2096577, 41.4841347, -17.5932388, 42.8905563, -60.1002121, 59.0773735

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_B1_B2_B2_A1_A2_A1

### Relational analysis result of IS_B2_B1_B2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5157223, upper bound: 57.5194923
time: 0.60 seconds

## Relational analysis of IS_B2_B1_B2_B2_A1_A2_A2

### Relational analysis result of IS_B2_B1_B2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5157223, upper bound: 57.5226226
time: 0.56 seconds

## BFS IS instance: IS_B2_B1_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -10.6360455, 44.2869492, -8.8608284, 37.6614952, -48.2975388, 53.1477776
1: -13.5139656, 50.1072121, -11.3173199, 42.6136665, -56.1276321, 61.4245300
2: -13.2801552, 49.8857727, -11.1280479, 42.2727051, -55.5528564, 61.0138206
3: -22.8301392, 53.2939072, -19.2408237, 45.4261627, -68.2563019, 72.5347290
4: -21.1995373, 51.3180428, -17.8677063, 43.5460930, -64.7456055, 69.1857300

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_B2_B1_B2_B2_A2_A1_A1

### Relational analysis result of IS_B2_B1_B2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4546673, upper bound: 57.4615284
time: 0.59 seconds

## Relational analysis of IS_B2_B1_B2_B2_A2_A1_A2

### Relational analysis result of IS_B2_B1_B2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4475735, upper bound: 57.4548957
time: 0.85 seconds

## BFS IS instance: IS_B2_B1_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -11.0944309, 46.1334305, -8.9674406, 38.0639191, -49.1583481, 55.1008720
1: -14.0975914, 52.1826591, -11.4526777, 43.0664177, -57.1640091, 63.6353378
2: -13.8261061, 51.9822998, -11.2560825, 42.7363510, -56.5624542, 63.2383804
3: -23.8071632, 55.4821091, -19.4629898, 45.9011917, -69.7083435, 74.9450989
4: -22.1109352, 53.4793167, -18.0676479, 44.0224457, -66.1333771, 71.5469437

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_B1_B2_B2_A2_A2_A1

### Relational analysis result of IS_B2_B1_B2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3785722, upper bound: 57.4375369
time: 0.59 seconds

## Relational analysis of IS_B2_B1_B2_B2_A2_A2_A2

### Relational analysis result of IS_B2_B1_B2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4168167, upper bound: 57.4621409
time: 0.61 seconds

## BFS IS instance: IS_B2_B2_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: -5.9786181, 27.7145119, -11.8374672, 48.4148979, -54.3935127, 39.5519676
1: -7.6296659, 31.5293980, -15.0364037, 54.8208542, -62.4505157, 46.5658035
2: -7.6836181, 30.7464123, -14.8005733, 54.6966438, -62.3802605, 45.5469818
3: -13.3743191, 33.7565308, -25.2378845, 58.3669090, -71.7412186, 58.9944153
4: -12.7490978, 31.5607872, -23.5150375, 56.6646614, -69.4137573, 55.0758247

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_B2_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_B2_B1_A1_A1_A1_B1

### Relational analysis result of IS_B2_B2_B1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4886127, upper bound: 57.5052602
time: 0.54 seconds

## Relational analysis of IS_B2_B2_B1_A1_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_B2_B1_A1_A1_A1_A1

### Relational analysis result of IS_B2_B2_B1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4686551, upper bound: 57.4271939
time: 0.63 seconds

## Relational analysis of IS_B2_B2_B1_A1_A1_A1_A2

### Relational analysis result of IS_B2_B2_B1_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4385113, upper bound: 57.4160443
time: 0.62 seconds

## BFS IS instance: IS_B2_B2_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -7.3111887, 32.6558228, -11.8374672, 48.4148979, -55.7260818, 44.4932785
1: -9.3543558, 37.0086670, -15.0364037, 54.8208542, -64.1752090, 52.0450706
2: -9.2846804, 36.4012794, -14.8005733, 54.6966438, -63.9813232, 51.2018509
3: -16.1798306, 39.5400085, -25.2378845, 58.3669090, -74.5467377, 64.7778931
4: -15.1875868, 37.3983688, -23.5150375, 56.6646614, -71.8522415, 60.9134064

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_B2_B1_A1_A1_A2_B1

### Relational analysis result of IS_B2_B2_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4886127, upper bound: 57.5052602
time: 0.68 seconds

## Relational analysis of IS_B2_B2_B1_A1_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_B2_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_B2_B2_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_B2_B2_B1_A1_A1_A2_B1

### Relational analysis result of IS_B2_B2_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5019055, upper bound: 57.5147807
time: 0.57 seconds

## Relational analysis of IS_B2_B2_B1_A1_A1_A2_B2

### Relational analysis result of IS_B2_B2_B1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4879846, upper bound: 57.4736466
time: 0.62 seconds

## BFS IS instance: IS_B2_B2_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -9.1041174, 38.8525696, -11.8374672, 48.4148979, -57.5190125, 50.6900253
1: -11.6246614, 44.0375786, -15.0364037, 54.8208542, -66.4455109, 59.0739822
2: -11.5010157, 43.5750198, -14.8005733, 54.6966438, -66.1976624, 58.3755836
3: -19.7765121, 47.0226326, -25.2378845, 58.3669090, -78.1434174, 72.2604980
4: -18.5012970, 44.9156151, -23.5150375, 56.6646614, -75.1659546, 68.4306488

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_B2_B1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_B2_B2_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B2_B2_B1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_B2_B1_A1_A2_A1_B1

### Relational analysis result of IS_B2_B2_B1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3726970, upper bound: 57.4577501
time: 0.62 seconds

## Relational analysis of IS_B2_B2_B1_A1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B2_B2_B1_A1_A2_A1_A1

### Relational analysis result of IS_B2_B2_B1_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4011074, upper bound: 57.4786907
time: 0.64 seconds

## Relational analysis of IS_B2_B2_B1_A1_A2_A1_A2

### Relational analysis result of IS_B2_B2_B1_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4011074, upper bound: 57.4944719
time: 0.62 seconds

## BFS IS instance: IS_B2_B2_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -9.7662954, 41.9631042, -11.8374672, 48.4148979, -58.1811790, 53.8005638
1: -12.4557714, 47.4921112, -15.0364037, 54.8208542, -67.2766037, 62.5285149
2: -12.3115005, 47.0263100, -14.8005733, 54.6966438, -67.0081406, 61.8268814
3: -21.2367153, 50.6065788, -25.2378845, 58.3669090, -79.6036224, 75.8444519
4: -19.8413486, 48.2990723, -23.5150375, 56.6646614, -76.5060043, 71.8141098

Time for backsubstitution: 2.29 seconds
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=66.57380676269531
rel_dist={0: [-57.5687467976788, 57.5687467976788]}

## Binary search (step 1) starts
Candidate diff: 0.1250000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5535923, upper bound: 57.5580881
time: 0.48 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5535923, upper bound: 57.5621985
time: 0.63 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.31 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.31
Output dim: 0, lower bound: -57.5535923, upper bound: 57.5580881
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.31
Output dim: 0, lower bound: -57.5535923, upper bound: 57.5621985

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -10.1770201, 42.4380188, -12.0890970, 49.7964020, -59.9734230, 54.5271149
1: -12.9492474, 48.0443954, -15.3444796, 56.3365631, -69.2858124, 63.3888741
2: -12.7204800, 47.7395439, -15.0430412, 56.2083435, -68.9288254, 62.7825813
3: -21.9290562, 51.2056084, -25.8391190, 59.8623886, -81.7914352, 77.0447235
4: -20.4464874, 49.1660233, -24.0019188, 57.9184074, -78.3648987, 73.1679230

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5512705, upper bound: 57.5512705
time: 0.55 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5512705, upper bound: 57.5580881
time: 0.57 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -11.9793425, 49.4708633, -13.0451412, 53.5286636, -65.5080032, 62.5160027
1: -15.2106190, 55.9559326, -16.5472050, 60.5470352, -75.7576523, 72.5031357
2: -14.9106045, 55.8369446, -16.2069473, 60.5032959, -75.4138870, 72.0438919
3: -25.6317139, 59.4555740, -27.8260193, 64.2862701, -89.9179688, 87.2815933
4: -23.7928352, 57.5261345, -25.8074226, 62.3490105, -86.1418457, 83.3335419

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5580881, upper bound: 57.5535923
time: 0.52 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5580881, upper bound: 57.5621985
time: 0.64 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.43 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.43
Output dim: 0, lower bound: -57.5512705, upper bound: 57.5512705
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.43
Output dim: 0, lower bound: -57.5512705, upper bound: 57.5580881
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.43
Output dim: 0, lower bound: -57.5580881, upper bound: 57.5535923
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.43
Output dim: 0, lower bound: -57.5580881, upper bound: 57.5621985

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -10.1770201, 42.4380188, -10.1770201, 42.4380188, -52.6150398, 52.6150398
1: -12.9492474, 48.0443954, -12.9492474, 48.0443954, -60.9936447, 60.9936447
2: -12.7204800, 47.7395439, -12.7204800, 47.7395439, -60.4600182, 60.4600143
3: -21.9290562, 51.2056084, -21.9290562, 51.2056084, -73.1346588, 73.1346588
4: -20.4464874, 49.1660233, -20.4464874, 49.1660233, -69.6124954, 69.6124954

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5131977, upper bound: 57.5301616
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5512705, upper bound: 57.5512705
time: 0.53 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -10.1770201, 42.4380188, -11.9793425, 49.4708633, -59.6478844, 54.4173622
1: -12.9492474, 48.0443954, -15.2106190, 55.9559326, -68.9051743, 63.2550125
2: -12.7204800, 47.7395439, -14.9106045, 55.8369446, -68.5574265, 62.6501427
3: -21.9290562, 51.2056084, -25.6317139, 59.4555740, -81.3846283, 76.8373108
4: -20.4464874, 49.1660233, -23.7928352, 57.5261345, -77.9726028, 72.9588547

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5311056, upper bound: 57.5395300
time: 0.51 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5497019, upper bound: 57.5573946
time: 0.56 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -11.9793425, 49.4708633, -10.1770201, 42.4380188, -54.4173622, 59.6478844
1: -15.2106190, 55.9559326, -12.9492474, 48.0443954, -63.2550125, 68.9051743
2: -14.9106045, 55.8369446, -12.7204800, 47.7395439, -62.6501465, 68.5574265
3: -25.6317139, 59.4555740, -21.9290562, 51.2056084, -76.8373108, 81.3846283
4: -23.7928352, 57.5261345, -20.4464874, 49.1660233, -72.9588547, 77.9725952

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4422203, upper bound: 57.3053738
time: 0.61 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5573946, upper bound: 57.5523846
time: 0.56 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -11.9793425, 49.4708633, -11.9793425, 49.4708633, -61.4502068, 61.4502068
1: -15.2106190, 55.9559326, -15.2106190, 55.9559326, -71.1665497, 71.1665497
2: -14.9106045, 55.8369446, -14.9106045, 55.8369446, -70.7475510, 70.7475510
3: -25.6317139, 59.4555740, -25.6317139, 59.4555740, -85.0872879, 85.0872879
4: -23.7928352, 57.5261345, -23.7928352, 57.5261345, -81.3189545, 81.3189621

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4785281, upper bound: 57.4907953
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4785281, upper bound: 57.4907953
time: 0.84 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.58 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.58
Output dim: 0, lower bound: -57.5131977, upper bound: 57.5301616
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.58
Output dim: 0, lower bound: -57.5512705, upper bound: 57.5512705
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.58
Output dim: 0, lower bound: -57.5311056, upper bound: 57.5395300
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.58
Output dim: 0, lower bound: -57.5497019, upper bound: 57.5573946
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.58
Output dim: 0, lower bound: -57.4422203, upper bound: 57.3053738
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.58
Output dim: 0, lower bound: -57.5573946, upper bound: 57.5523846
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.58
Output dim: 0, lower bound: -57.4785281, upper bound: 57.4907953
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.58
Output dim: 0, lower bound: -57.4785281, upper bound: 57.4907953

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -9.1575928, 38.8744278, -9.5324383, 40.0096436, -49.1672325, 48.4068680
1: -11.6959438, 44.0212021, -12.1431637, 45.3053627, -57.0013046, 56.1643677
2: -11.4683819, 43.6604576, -11.9324455, 44.9406128, -56.4089966, 55.5929031
3: -19.9659348, 46.8748016, -20.6196671, 48.3145409, -68.2804718, 67.4944534
4: -18.4506874, 44.9865723, -19.2304115, 46.2580605, -64.7087479, 64.2169800

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4527510, upper bound: 57.5004519
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A1_A2

### Relational analysis result of IS_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5095796, upper bound: 57.5210587
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8.8118200, 37.3577957, -10.1770201, 42.4380188, -51.2498398, 47.5348167
1: -11.2442751, 42.3229713, -12.9492474, 48.0443954, -59.2886696, 55.2722168
2: -11.0679474, 41.8824844, -12.7204800, 47.7395439, -58.8074799, 54.6029587
3: -19.1606712, 45.1869125, -21.9290562, 51.2056084, -70.3662796, 67.1159668
4: -17.8871288, 43.1344337, -20.4464874, 49.1660233, -67.0531387, 63.5809097

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5340187, upper bound: 57.5226460
time: 0.51 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5340187, upper bound: 57.5512705
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -9.1517153, 39.0846024, -11.1019211, 46.1585426, -55.3102570, 50.1865234
1: -11.7220478, 44.2469559, -14.1076412, 52.2187729, -63.9408188, 58.3545990
2: -11.4693985, 43.8899803, -13.8358870, 52.0074997, -63.4768867, 57.7258606
3: -20.1092834, 47.1997299, -23.8519135, 55.5298004, -75.6390839, 71.0516434
4: -18.6677227, 45.1985092, -22.1600800, 53.4977493, -72.1654739, 67.3585815

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5030340, upper bound: 57.4835618
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5168636, upper bound: 57.5260394
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -9.8234758, 41.2086678, -11.9793425, 49.4708633, -59.2943306, 53.1880074
1: -12.5056238, 46.6563835, -15.2106190, 55.9559326, -68.4615479, 61.8670044
2: -12.2946529, 46.3034515, -14.9106045, 55.8369446, -68.1315994, 61.2140579
3: -21.2287254, 49.7448921, -25.6317139, 59.4555740, -80.6842957, 75.3765793
4: -19.8178635, 47.6446228, -23.7928352, 57.5261345, -77.3439789, 71.4374542

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5112195, upper bound: 57.5284482
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5523846, upper bound: 57.5573946
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -11.1019211, 46.1585426, -9.1517153, 39.0846024, -50.1865234, 55.3102570
1: -14.1076412, 52.2187729, -11.7220478, 44.2469559, -58.3545990, 63.9408150
2: -13.8358870, 52.0074997, -11.4693985, 43.8899803, -57.7258682, 63.4768867
3: -23.8519135, 55.5298004, -20.1092834, 47.1997299, -71.0516434, 75.6390839
4: -22.1600800, 53.4977493, -18.6677227, 45.1985092, -67.3585815, 72.1654739

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4835618, upper bound: 57.5030340
time: 0.54 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5260394, upper bound: 57.5168636
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -11.9793425, 49.4708633, -9.8234758, 41.2086678, -53.1880074, 59.2943306
1: -15.2106190, 55.9559326, -12.5056238, 46.6563835, -61.8670044, 68.4615479
2: -14.9106045, 55.8369446, -12.2946529, 46.3034515, -61.2140541, 68.1315994
3: -25.6317139, 59.4555740, -21.2287254, 49.7448921, -75.3765793, 80.6842957
4: -23.7928352, 57.5261345, -19.8178635, 47.6446228, -71.4374542, 77.3439865

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5284482, upper bound: 57.5112195
time: 0.56 seconds

## Relational analysis of IS_A2_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5573946, upper bound: 57.5523846
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -11.0297852, 46.1416092, -11.2561569, 46.7493057, -57.7790871, 57.3977623
1: -14.0756989, 52.2037125, -14.3058786, 52.8868065, -66.9625092, 66.5095901
2: -13.7119274, 52.0922241, -14.0253191, 52.7105789, -66.4224854, 66.1175461
3: -23.8188839, 55.4197121, -24.1612244, 56.2217560, -80.0406418, 79.5809326
4: -21.8664837, 53.5777702, -22.4216366, 54.2697601, -76.1362457, 75.9994049

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A1_A1

### Relational analysis result of IS_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.1381061, upper bound: 57.3156531
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A1_A2

### Relational analysis result of IS_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4781048, upper bound: 57.4904413
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -10.4046993, 43.4960098, -11.9793425, 49.4708633, -59.8755646, 55.4753494
1: -13.2431650, 49.2144547, -15.2106190, 55.9559326, -69.1990967, 64.4250717
2: -12.9902172, 48.9800606, -14.9106045, 55.8369446, -68.8271637, 63.8906631
3: -22.4255047, 52.3740425, -25.6317139, 59.4555740, -81.8810806, 78.0057449
4: -20.8130035, 50.4081459, -23.7928352, 57.5261345, -78.3391266, 74.2009735

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5020090, upper bound: 57.4820486
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4907953, upper bound: 57.5621587
time: 0.59 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.42 seconds
IS_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 3.42
Output dim: 0, lower bound: -57.4527510, upper bound: 57.5004519
IS_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.42
Output dim: 0, lower bound: -57.5095796, upper bound: 57.5210587
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.42
Output dim: 0, lower bound: -57.5340187, upper bound: 57.5226460
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.42
Output dim: 0, lower bound: -57.5340187, upper bound: 57.5512705
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.42
Output dim: 0, lower bound: -57.5030340, upper bound: 57.4835618
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.42
Output dim: 0, lower bound: -57.5168636, upper bound: 57.5260394
IS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.42
Output dim: 0, lower bound: -57.5112195, upper bound: 57.5284482
IS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.42
Output dim: 0, lower bound: -57.5523846, upper bound: 57.5573946
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.42
Output dim: 0, lower bound: -57.4835618, upper bound: 57.5030340
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.42
Output dim: 0, lower bound: -57.5260394, upper bound: 57.5168636
IS_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.42
Output dim: 0, lower bound: -57.5284482, upper bound: 57.5112195
IS_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.42
Output dim: 0, lower bound: -57.5573946, upper bound: 57.5523846
IS_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 3.42
Output dim: 0, lower bound: -57.1381061, upper bound: 57.3156531
IS_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.42
Output dim: 0, lower bound: -57.4781048, upper bound: 57.4904413
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.42
Output dim: 0, lower bound: -57.5020090, upper bound: 57.4820486
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.42
Output dim: 0, lower bound: -57.4907953, upper bound: 57.5621587

## BFS IS instance: IS_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -6.2563448, 28.6933155, -9.5324383, 40.0096436, -46.2659874, 38.2257538
1: -7.9915113, 32.5977592, -12.1431637, 45.3053627, -53.2968674, 44.7409172
2: -7.9890079, 31.8166008, -11.9324455, 44.9406128, -52.9296150, 43.7490425
3: -14.0460072, 34.9063377, -20.6196671, 48.3145409, -62.3605499, 55.5260048
4: -13.2832928, 32.6852341, -19.2304115, 46.2580605, -59.5413399, 51.9156456

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4507852, upper bound: 57.4989698
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_A1_A2

### Relational analysis result of IS_A1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4439877, upper bound: 57.4972253
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -7.4350624, 32.4608459, -8.5959616, 36.4160042, -43.8510666, 41.0568008
1: -9.5189266, 36.8422890, -10.9633245, 41.2568817, -50.7758102, 47.8056145
2: -9.4125261, 36.2255974, -10.8006697, 40.8098907, -50.2224159, 47.0262680
3: -16.3478165, 39.4466934, -18.6705551, 44.0664062, -60.4142113, 58.1172485
4: -15.2531500, 37.2985878, -17.4567699, 42.0210190, -57.2741661, 54.7553558

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_A2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5061430, upper bound: 57.5061431
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5061430, upper bound: 57.5210587
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.8118200, 37.3577957, -9.1575928, 38.8744278, -47.6862488, 46.5153885
1: -11.2442751, 42.3229713, -11.6959438, 44.0212021, -55.2654762, 54.0189133
2: -11.0679474, 41.8824844, -11.4683819, 43.6604576, -54.7284012, 53.3508682
3: -19.1606712, 45.1869125, -19.9659348, 46.8748016, -66.0354691, 65.1528397
4: -17.8871288, 43.1344337, -18.4506874, 44.9865723, -62.8737030, 61.5851212

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5004519, upper bound: 57.4527510
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5210587, upper bound: 57.5095796
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.8118200, 37.3577957, -8.8118200, 37.3577957, -46.1696167, 46.1696167
1: -11.2442751, 42.3229713, -11.2442751, 42.3229713, -53.5672455, 53.5672455
2: -11.0679474, 41.8824844, -11.0679474, 41.8824844, -52.9504280, 52.9504280
3: -19.1606712, 45.1869125, -19.1606712, 45.1869125, -64.3475800, 64.3475800
4: -17.8871288, 43.1344337, -17.8871288, 43.1344337, -61.0215607, 61.0215607

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5004519, upper bound: 57.4945006
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5210587, upper bound: 57.5326214
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.1517153, 39.0846024, -8.0074329, 35.2956505, -44.4473648, 47.0920334
1: -11.7220478, 44.2469559, -10.2124548, 39.9718704, -51.6939163, 54.4594116
2: -11.4693985, 43.8899803, -10.1203489, 39.3907547, -50.8601532, 54.0103226
3: -20.1092834, 47.1997299, -17.6419621, 42.6744690, -62.7837524, 64.8416901
4: -18.6677227, 45.1985092, -16.5761509, 40.4280548, -59.0957756, 61.7746506

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4044075, upper bound: 57.2659297
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2

### Relational analysis result of IS_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4044075, upper bound: 57.4835618
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -8.5184622, 36.6874084, -9.0371084, 38.3006172, -46.8190804, 45.7245140
1: -10.9145231, 41.5507507, -11.5198345, 43.3769875, -54.2915077, 53.0705872
2: -10.7067871, 41.1160088, -11.3451462, 42.9124069, -53.6191940, 52.4611511
3: -18.7815094, 44.3796196, -19.5400734, 46.3099709, -65.0914764, 63.9196930
4: -17.4920273, 42.3493729, -18.2220364, 44.1050758, -61.5970993, 60.5713997

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4947398, upper bound: 57.5011140
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5168636, upper bound: 57.5260394
time: 1.07 seconds

## BFS IS instance: IS_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -8.8402529, 37.7630539, -11.2561569, 46.7493057, -55.5895576, 49.0192070
1: -11.2974596, 42.7681274, -14.3058786, 52.8868065, -64.1842499, 57.0740051
2: -11.0862007, 42.3602257, -14.0253191, 52.7105789, -63.7967682, 56.3855438
3: -19.3324356, 45.5596771, -24.1612244, 56.2217560, -75.5541916, 69.7208939
4: -17.8893318, 43.6359253, -22.4216366, 54.2697601, -72.1590805, 66.0575638

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_A1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4817997, upper bound: 57.4696927
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4969975, upper bound: 57.5155763
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -8.4693804, 36.1834831, -11.9793425, 49.4708633, -57.9402351, 48.1628227
1: -10.8127460, 41.0070839, -15.2106190, 55.9559326, -66.7686615, 56.2177048
2: -10.6570034, 40.5084305, -14.9106045, 55.8369446, -66.4939499, 55.4190369
3: -18.4758568, 43.8035660, -25.6317139, 59.4555740, -77.9314270, 69.4352722
4: -17.2884102, 41.7039909, -23.7928352, 57.5261345, -74.8145294, 65.4968262

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5259404, upper bound: 57.5126437
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2

### Relational analysis result of IS_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5339194, upper bound: 57.5369324
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -8.0074329, 35.2956505, -9.1517153, 39.0846024, -47.0920296, 44.4473648
1: -10.2124548, 39.9718704, -11.7220478, 44.2469559, -54.4594116, 51.6939163
2: -10.1203489, 39.3907547, -11.4693985, 43.8899803, -54.0103226, 50.8601532
3: -17.6419621, 42.6744690, -20.1092834, 47.1997299, -64.8416901, 62.7837524
4: -16.5761509, 40.4280548, -18.6677227, 45.1985092, -61.7746506, 59.0957756

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2659297, upper bound: 57.4044075
time: 0.60 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2

### Relational analysis result of IS_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2659297, upper bound: 57.5030340
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -9.0371084, 38.3006172, -8.5184622, 36.6874084, -45.7245140, 46.8190804
1: -11.5198345, 43.3769875, -10.9145231, 41.5507507, -53.0705872, 54.2915077
2: -11.3451462, 42.9124069, -10.7067871, 41.1160088, -52.4611511, 53.6191940
3: -19.5400734, 46.3099709, -18.7815094, 44.3796196, -63.9196930, 65.0914764
4: -18.2220364, 44.1050758, -17.4920273, 42.3493729, -60.5713997, 61.5971031

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5011140, upper bound: 57.4947398
time: 0.53 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5260394, upper bound: 57.5168636
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -11.2561569, 46.7493057, -8.8402529, 37.7630539, -49.0192070, 55.5895538
1: -14.3058786, 52.8868065, -11.2974596, 42.7681274, -57.0740051, 64.1842575
2: -14.0253191, 52.7105789, -11.0862007, 42.3602257, -56.3855438, 63.7967644
3: -24.1612244, 56.2217560, -19.3324356, 45.5596771, -69.7209015, 75.5541916
4: -22.4216366, 54.2697601, -17.8893318, 43.6359253, -66.0575638, 72.1590805

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_B2_B1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4696927, upper bound: 57.4817997
time: 0.58 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5155763, upper bound: 57.4969975
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -11.9793425, 49.4708633, -8.4693804, 36.1834831, -48.1628227, 57.9402390
1: -15.2106190, 55.9559326, -10.8127460, 41.0070839, -56.2177048, 66.7686615
2: -14.9106045, 55.8369446, -10.6570034, 40.5084305, -55.4190331, 66.4939499
3: -25.6317139, 59.4555740, -18.4758568, 43.8035660, -69.4352722, 77.9314270
4: -23.7928352, 57.5261345, -17.2884102, 41.7039909, -65.4968262, 74.8145370

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_B2_B2_A1

### Relational analysis result of IS_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4937896, upper bound: 57.5259404
time: 0.62 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2

### Relational analysis result of IS_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5369324, upper bound: 57.5339194
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -11.0847101, 47.3818054, -10.3802204, 43.4340286, -54.5187340, 57.7620087
1: -14.2560978, 53.6242714, -13.2013044, 49.1462631, -63.4023514, 66.8255768
2: -13.7948971, 53.5095329, -12.9549942, 48.8752823, -62.6701813, 66.4645233
3: -24.3539143, 56.8776131, -22.3787231, 52.2973862, -76.6512985, 79.2563324
4: -22.2534161, 54.9756432, -20.7991161, 50.2737999, -72.5272141, 75.7747574

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_B2_A1_A1_A1

### Relational analysis result of IS_A2_B2_A1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -57.0956065, upper bound: 57.2771349
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A1_A1_B1

### Relational analysis result of IS_A2_B2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -57.0553677, upper bound: 57.1244992
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A1_A1_B2

### Relational analysis result of IS_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.0553677, upper bound: 57.3156531
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -10.6957531, 44.9465866, -11.2561569, 46.7493057, -57.4450531, 56.2027397
1: -13.6549959, 50.8562546, -14.3058786, 52.8868065, -66.5417862, 65.1621323
2: -13.3063459, 50.7025528, -14.0253191, 52.7105789, -66.0169220, 64.7278442
3: -23.1477661, 54.0001144, -24.1612244, 56.2217560, -79.3695221, 78.1613312
4: -21.2595654, 52.1365585, -22.4216366, 54.2697601, -75.5293274, 74.5581970

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A1_A2_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -57.2091961, upper bound: 57.1524313
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2

### Relational analysis result of IS_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2091961, upper bound: 57.4904413
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -10.4046993, 43.4960098, -11.0297852, 46.1416092, -56.5463028, 54.5257950
1: -13.2431650, 49.2144547, -14.0756989, 52.2037125, -65.4468765, 63.2901535
2: -12.9902172, 48.9800606, -13.7119274, 52.0922241, -65.0824432, 62.6919861
3: -22.4255047, 52.3740425, -23.8188839, 55.4197121, -77.8452148, 76.1929245
4: -20.8130035, 50.4081459, -21.8664837, 53.5777702, -74.3907700, 72.2746277

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -57.1524313, upper bound: 57.2091961
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5016550, upper bound: 57.4817208
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -10.4046993, 43.4960098, -10.4046993, 43.4960098, -53.9007034, 53.9006996
1: -13.2431650, 49.2144547, -13.2431650, 49.2144547, -62.4576187, 62.4576187
2: -12.9902172, 48.9800606, -12.9902172, 48.9800606, -61.9702759, 61.9702644
3: -22.4255047, 52.3740425, -22.4255047, 52.3740425, -74.7995453, 74.7995453
4: -20.8130035, 50.4081459, -20.8130035, 50.4081459, -71.2211304, 71.2211304

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.1524313, upper bound: 57.3737716
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5016550, upper bound: 57.5620557
time: 0.91 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.91 seconds
IS_A1_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.91
Output dim: 0, lower bound: -57.4507852, upper bound: 57.4989698
IS_A1_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.91
Output dim: 0, lower bound: -57.4439877, upper bound: 57.4972253
IS_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.91
Output dim: 0, lower bound: -57.5061430, upper bound: 57.5061431
IS_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.91
Output dim: 0, lower bound: -57.5061430, upper bound: 57.5210587
IS_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.91
Output dim: 0, lower bound: -57.5004519, upper bound: 57.4527510
IS_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.91
Output dim: 0, lower bound: -57.5210587, upper bound: 57.5095796
IS_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.91
Output dim: 0, lower bound: -57.5004519, upper bound: 57.4945006
IS_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.91
Output dim: 0, lower bound: -57.5210587, upper bound: 57.5326214
IS_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.91
Output dim: 0, lower bound: -57.4044075, upper bound: 57.2659297
IS_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.91
Output dim: 0, lower bound: -57.4044075, upper bound: 57.4835618
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.91
Output dim: 0, lower bound: -57.4947398, upper bound: 57.5011140
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.91
Output dim: 0, lower bound: -57.5168636, upper bound: 57.5260394
IS_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.91
Output dim: 0, lower bound: -57.4817997, upper bound: 57.4696927
IS_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.91
Output dim: 0, lower bound: -57.4969975, upper bound: 57.5155763
IS_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.91
Output dim: 0, lower bound: -57.5259404, upper bound: 57.5126437
IS_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.91
Output dim: 0, lower bound: -57.5339194, upper bound: 57.5369324
IS_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.91
Output dim: 0, lower bound: -57.2659297, upper bound: 57.4044075
IS_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.91
Output dim: 0, lower bound: -57.2659297, upper bound: 57.5030340
IS_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.91
Output dim: 0, lower bound: -57.5011140, upper bound: 57.4947398
IS_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.91
Output dim: 0, lower bound: -57.5260394, upper bound: 57.5168636
IS_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.91
Output dim: 0, lower bound: -57.4696927, upper bound: 57.4817997
IS_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.91
Output dim: 0, lower bound: -57.5155763, upper bound: 57.4969975
IS_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.91
Output dim: 0, lower bound: -57.4937896, upper bound: 57.5259404
IS_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.91
Output dim: 0, lower bound: -57.5369324, upper bound: 57.5339194
IS_A2_B2_A1_A1_B1, status: Status.VERIFIED, split count: 5, time: 3.91
Output dim: 0, lower bound: -57.0553677, upper bound: 57.1244992
IS_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.91
Output dim: 0, lower bound: -57.0553677, upper bound: 57.3156531
IS_A2_B2_A1_A2_B1, status: Status.VERIFIED, split count: 5, time: 3.91
Output dim: 0, lower bound: -57.2091961, upper bound: 57.1524313
IS_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.91
Output dim: 0, lower bound: -57.2091961, upper bound: 57.4904413
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.91
Output dim: 0, lower bound: -57.1524313, upper bound: 57.2091961
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.91
Output dim: 0, lower bound: -57.5016550, upper bound: 57.4817208
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.91
Output dim: 0, lower bound: -57.1524313, upper bound: 57.3737716
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.91
Output dim: 0, lower bound: -57.5016550, upper bound: 57.5620557

## BFS IS instance: IS_A1_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: -5.5928502, 27.0996017, -8.6625891, 36.7026978, -42.2955475, 35.7621803
1: -7.1515408, 30.8633766, -11.0442200, 41.5776787, -48.7292175, 41.9075928
2: -7.1898627, 29.9638920, -10.8752861, 41.1112785, -48.3011398, 40.8391762
3: -12.8735600, 32.9426193, -18.8350601, 44.4135475, -57.2871094, 51.7776794
4: -12.2557392, 30.6781273, -17.6368332, 42.3082314, -54.5639687, 48.3149490

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_A1_A1_B1

### Relational analysis result of IS_A1_B1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4350126, upper bound: 57.4527563
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A1_A1_A1_B2

### Relational analysis result of IS_A1_B1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4350126, upper bound: 57.4989698
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -5.9486384, 27.6570396, -9.5324383, 40.0096436, -45.9582825, 37.1894760
1: -7.5847530, 31.4490318, -12.1431637, 45.3053627, -52.8901138, 43.5921936
2: -7.6243067, 30.6088829, -11.9324455, 44.9406128, -52.5649185, 42.5413284
3: -13.3943548, 33.6805496, -20.6196671, 48.3145409, -61.7088890, 54.3002129
4: -12.7465086, 31.4047012, -19.2304115, 46.2580605, -59.0045662, 50.6351128

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_A1_A2_B1

### Relational analysis result of IS_A1_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4295416, upper bound: 57.4545045
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A1_A1_A2_B2

### Relational analysis result of IS_A1_B1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4295416, upper bound: 57.4972253
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -7.4350624, 32.4608459, -7.9151025, 33.9736404, -41.4087029, 40.3759346
1: -9.5189266, 36.8422890, -10.1209154, 38.5097313, -48.0286560, 46.9632034
2: -9.4125261, 36.2255974, -9.9639902, 38.0409660, -47.4534912, 46.1895866
3: -16.3478165, 39.4466934, -17.3410969, 41.1127625, -57.4605789, 56.7877884
4: -15.2531500, 37.2985878, -16.1062527, 39.2171516, -54.4703026, 53.4048386

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4859338, upper bound: 57.4494298
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4859338, upper bound: 57.5061026
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -7.4350624, 32.4608459, -7.9333544, 34.0408020, -41.4758644, 40.3941994
1: -9.5189266, 36.8422890, -10.1330433, 38.6041336, -48.1230621, 46.9753304
2: -9.4125261, 36.2255974, -10.0094357, 38.0686302, -47.4811554, 46.2350311
3: -16.3478165, 39.4466934, -17.3285427, 41.2797546, -57.6275673, 56.7752304
4: -15.2531500, 37.2985878, -16.2363853, 39.2171059, -54.4702568, 53.5349731

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4859338, upper bound: 57.4771794
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4859338, upper bound: 57.5061026
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -8.8118200, 37.3577957, -6.2563448, 28.6933155, -37.5051346, 43.6141396
1: -11.2442751, 42.3229713, -7.9915113, 32.5977592, -43.8420258, 50.3144836
2: -11.0679474, 41.8824844, -7.9890079, 31.8166008, -42.8845406, 49.8714867
3: -19.1606712, 45.1869125, -14.0460072, 34.9063377, -54.0670052, 59.2329178
4: -17.8871288, 43.1344337, -13.2832928, 32.6852341, -50.5723648, 56.4177208

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B1_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4989698, upper bound: 57.4507852
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4972253, upper bound: 57.4439877
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -7.9333544, 34.0408020, -7.4350624, 32.4608459, -40.3941994, 41.4758644
1: -10.1330433, 38.6041336, -9.5189266, 36.8422890, -46.9753304, 48.1230621
2: -10.0094357, 38.0686302, -9.4125261, 36.2255974, -46.2350311, 47.4811554
3: -17.3285427, 41.2797546, -16.3478165, 39.4466934, -56.7752304, 57.6275711
4: -16.2363853, 39.2171059, -15.2531500, 37.2985878, -53.5349731, 54.4702568

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5210587, upper bound: 57.5094799
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5210587, upper bound: 57.5095796
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -8.8118200, 37.3577957, -5.9096947, 27.3499146, -36.1617317, 43.2674904
1: -11.2442751, 42.3229713, -7.5164185, 31.1457329, -42.3900032, 49.8393898
2: -11.0679474, 41.8824844, -7.6001306, 30.2378502, -41.3057938, 49.4826164
3: -19.1606712, 45.1869125, -13.2140646, 33.4282455, -52.5889130, 58.4009743
4: -17.8871288, 43.1344337, -12.7368975, 30.9877415, -48.8748703, 55.8713303

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5092668, upper bound: 57.4692860
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A2_B2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5186089, upper bound: 57.4937896
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -7.9333544, 34.0408020, -7.3467827, 32.0613708, -39.9947243, 41.3875809
1: -10.1330433, 38.6041336, -9.3942413, 36.4329567, -46.5659904, 47.9983749
2: -10.0094357, 38.0686302, -9.3245058, 35.7049446, -45.7143784, 47.3931351
3: -17.3285427, 41.2797546, -16.1041088, 39.0988007, -56.4273338, 57.3838615
4: -16.2363853, 39.2171059, -15.1873417, 36.7418594, -52.9782410, 54.4044495

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_B2_B1

### Relational analysis result of IS_A1_B1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5326215, upper bound: 57.5326214
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2_B2

### Relational analysis result of IS_A1_B1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5326215, upper bound: 57.5326214
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -9.1517153, 39.0846024, -7.9552212, 35.8122635, -44.9639778, 47.0398178
1: -11.7220478, 44.2469559, -10.2020988, 40.5423241, -52.2643738, 54.4490547
2: -11.4693985, 43.8899803, -10.0640669, 40.0233917, -51.4927902, 53.9540482
3: -20.1092834, 47.1997299, -17.8135185, 43.2676315, -63.3769150, 65.0132446
4: -18.6677227, 45.1985092, -16.6815376, 41.0891838, -59.7569046, 61.8800354

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3971059, upper bound: 57.2644628
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A1_B1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4044075, upper bound: 57.2659297
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -9.1517153, 39.0846024, -8.4519434, 37.0708733, -46.2225876, 47.5365410
1: -11.7220478, 44.2469559, -10.7818136, 41.9590607, -53.6811066, 55.0287704
2: -11.4693985, 43.8899803, -10.6651726, 41.4331932, -52.9025917, 54.5551529
3: -20.1092834, 47.1997299, -18.5733147, 44.7520294, -64.8613129, 65.7730408
4: -18.6677227, 45.1985092, -17.3966503, 42.5266571, -61.1943817, 62.5951614

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3564297, upper bound: 57.4670178
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3564297, upper bound: 57.4835618
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -7.3018093, 32.6713791, -8.4117498, 35.9554672, -43.2572746, 41.0831261
1: -9.3889122, 37.0353622, -10.7277308, 40.7539024, -50.1428146, 47.7630844
2: -9.2342148, 36.5235863, -10.5861588, 40.2136612, -49.4478760, 47.1097450
3: -16.3975430, 39.5195465, -18.2507763, 43.5593109, -59.9568558, 57.7703209
4: -15.2220888, 37.6156197, -17.0741596, 41.3301773, -56.5522652, 54.6897812

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4226292, upper bound: 57.4776458
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4226292, upper bound: 57.4939629
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -7.4394846, 32.9472542, -9.0371084, 38.3006172, -45.7401009, 41.9843597
1: -9.5418024, 37.3610992, -11.5198345, 43.3769875, -52.9187889, 48.8809357
2: -9.4033060, 36.7727203, -11.3451462, 42.9124069, -52.3157082, 48.1178665
3: -16.5861855, 39.9446564, -19.5400734, 46.3099709, -62.8961563, 59.4847260
4: -15.5030107, 37.8687439, -18.2220364, 44.1050758, -59.6080780, 56.0907745

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4551317, upper bound: 57.5093507
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4551317, upper bound: 57.5199713
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -8.8402529, 37.7630539, -8.2398510, 36.1620255, -45.0022774, 46.0028992
1: -11.2974596, 42.7681274, -10.5217819, 40.9423981, -52.2398567, 53.2899094
2: -11.0862007, 42.3602257, -10.4027452, 40.4170303, -51.5032310, 52.7629700
3: -19.3324356, 45.5596771, -18.1303902, 43.6821404, -63.0145760, 63.6900673
4: -17.8893318, 43.6359253, -16.9685364, 41.5019875, -59.3913155, 60.6044617

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4227620, upper bound: 57.4464805
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4227620, upper bound: 57.4696927
time: 0.94 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -7.7769594, 33.7355652, -9.2360849, 38.9808388, -46.7577972, 42.9716415
1: -9.9515686, 38.2421722, -11.7805223, 44.1449623, -54.0965309, 50.0226822
2: -9.8044128, 37.7289734, -11.5826206, 43.7237854, -53.5281982, 49.3115921
3: -17.1150990, 40.8141479, -19.9495296, 47.1118469, -64.2269440, 60.7636719
4: -15.9046125, 38.8731270, -18.5434914, 44.9581947, -60.8628082, 57.4166183

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4969975, upper bound: 57.5155763
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4961385, upper bound: 57.5115221
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -8.4693804, 36.1834831, -8.8355389, 38.4204979, -46.8898735, 45.0190201
1: -10.8127460, 41.0070839, -11.2725945, 43.4727173, -54.2854614, 52.2796783
2: -10.6570034, 40.5084305, -11.1283092, 43.0069008, -53.6639023, 51.6367416
3: -18.4758568, 43.8035660, -19.3558121, 46.3429070, -64.8187637, 63.1593666
4: -17.2884102, 41.7039909, -18.0807438, 44.1607170, -61.4491272, 59.7847328

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_A2_B1_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4630019, upper bound: 57.4096086
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_B2

### Relational analysis result of IS_A1_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4630019, upper bound: 57.4986537
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -7.5869489, 32.8869972, -9.8936510, 41.4662666, -49.0532150, 42.7806473
1: -9.6928911, 37.3132858, -12.6075249, 46.9325066, -56.6253967, 49.9208107
2: -9.5936756, 36.7145805, -12.3815975, 46.5865746, -56.1802521, 49.0961761
3: -16.6362324, 39.9200401, -21.2981529, 50.0432014, -66.6794357, 61.2181931
4: -15.6380339, 37.8018875, -19.7658882, 47.8902588, -63.5282860, 57.5677605

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4732274, upper bound: 57.4617319
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_B2

### Relational analysis result of IS_A1_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4732274, upper bound: 57.5315951
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -7.9552212, 35.8122635, -9.1517153, 39.0846024, -47.0398216, 44.9639778
1: -10.2020988, 40.5423241, -11.7220478, 44.2469559, -54.4490547, 52.2643738
2: -10.0640669, 40.0233917, -11.4693985, 43.8899803, -53.9540482, 51.4927902
3: -17.8135185, 43.2676315, -20.1092834, 47.1997299, -65.0132446, 63.3769150
4: -16.6815376, 41.0891838, -18.6677227, 45.1985092, -61.8800354, 59.7568970

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B1_A1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2644628, upper bound: 57.3971059
time: 0.53 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2659297, upper bound: 57.4044075
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -8.4519434, 37.0708733, -9.1517153, 39.0846024, -47.5365410, 46.2225876
1: -10.7818136, 41.9590607, -11.7220478, 44.2469559, -55.0287704, 53.6811066
2: -10.6651726, 41.4331932, -11.4693985, 43.8899803, -54.5551529, 52.9025917
3: -18.5733147, 44.7520294, -20.1092834, 47.1997299, -65.7730408, 64.8613129
4: -17.3966503, 42.5266571, -18.6677227, 45.1985092, -62.5951614, 61.1943817

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_B1_A1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2491961, upper bound: 57.4551317
time: 0.60 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2491961, upper bound: 57.5030340
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.4117498, 35.9554672, -7.3018093, 32.6713791, -41.0831261, 43.2572784
1: -10.7277308, 40.7539024, -9.3889122, 37.0353622, -47.7630844, 50.1428146
2: -10.5861588, 40.2136612, -9.2342148, 36.5235863, -47.1097450, 49.4478760
3: -18.2507763, 43.5593109, -16.3975430, 39.5195465, -57.7703133, 59.9568520
4: -17.0741596, 41.3301773, -15.2220888, 37.6156197, -54.6897812, 56.5522652

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_B1_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4776459, upper bound: 57.4352799
time: 0.54 seconds

## Relational analysis of IS_A2_B1_B1_A2_B1_B2

### Relational analysis result of IS_A2_B1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4776459, upper bound: 57.4947398
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.0371084, 38.3006172, -7.4394846, 32.9472542, -41.9843597, 45.7401009
1: -11.5198345, 43.3769875, -9.5418024, 37.3610992, -48.8809357, 52.9187889
2: -11.3451462, 42.9124069, -9.4033060, 36.7727203, -48.1178665, 52.3157120
3: -19.5400734, 46.3099709, -16.5861855, 39.9446564, -59.4847260, 62.8961563
4: -18.2220364, 44.1050758, -15.5030107, 37.8687439, -56.0907745, 59.6080666

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_B1_A2_B2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5093507, upper bound: 57.4689063
time: 0.56 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_B2

### Relational analysis result of IS_A2_B1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5093508, upper bound: 57.5168636
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -8.2398510, 36.1620255, -8.8402529, 37.7630539, -46.0028954, 45.0022774
1: -10.5217819, 40.9423981, -11.2974596, 42.7681274, -53.2899094, 52.2398567
2: -10.4027452, 40.4170303, -11.0862007, 42.3602257, -52.7629700, 51.5032310
3: -18.1303902, 43.6821404, -19.3324356, 45.5596771, -63.6900673, 63.0145760
4: -16.9685364, 41.5019875, -17.8893318, 43.6359253, -60.6044617, 59.3913155

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_B2_B1_A1_B1

### Relational analysis result of IS_A2_B1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4464805, upper bound: 57.4227620
time: 0.61 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_B2

### Relational analysis result of IS_A2_B1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4464805, upper bound: 57.4817997
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -9.2360849, 38.9808388, -7.7769594, 33.7355652, -42.9716415, 46.7577972
1: -11.7805223, 44.1449623, -9.9515686, 38.2421722, -50.0226822, 54.0965309
2: -11.5826206, 43.7237854, -9.8044128, 37.7289734, -49.3115921, 53.5281982
3: -19.9495296, 47.1118469, -17.1150990, 40.8141479, -60.7636642, 64.2269440
4: -18.5434914, 44.9581947, -15.9046125, 38.8731270, -57.4166183, 60.8628082

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_B2_B1_A2_A1

### Relational analysis result of IS_A2_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5155763, upper bound: 57.4969975
time: 1.14 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2_A2

### Relational analysis result of IS_A2_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4696927, upper bound: 57.4817997
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -8.8355389, 38.4204979, -8.4693804, 36.1834831, -45.0190201, 46.8898735
1: -11.2725945, 43.4727173, -10.8127460, 41.0070839, -52.2796783, 54.2854614
2: -11.1283092, 43.0069008, -10.6570034, 40.5084305, -51.6367416, 53.6639023
3: -19.3558121, 46.3429070, -18.4758568, 43.8035660, -63.1593666, 64.8187637
4: -18.0807438, 44.1607170, -17.2884102, 41.7039909, -59.7847290, 61.4491272

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4063586, upper bound: 57.4630019
time: 0.73 seconds

## Relational analysis of IS_A2_B1_B2_B2_A1_A2

### Relational analysis result of IS_A2_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4063586, upper bound: 57.5259404
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -9.8936510, 41.4662666, -7.5869489, 32.8869972, -42.7806473, 49.0532150
1: -12.6075249, 46.9325066, -9.6928911, 37.3132858, -49.9208107, 56.6253967
2: -12.3815975, 46.5865746, -9.5936756, 36.7145805, -49.0961761, 56.1802521
3: -21.2981529, 50.0432014, -16.6362324, 39.9200401, -61.2181892, 66.6794357
4: -19.7658882, 47.8902588, -15.6380339, 37.8018875, -57.5677757, 63.5282936

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_B2_A2_A1

### Relational analysis result of IS_A2_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4593124, upper bound: 57.4732274
time: 0.61 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2_A2

### Relational analysis result of IS_A2_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4593124, upper bound: 57.5339194
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -11.0847101, 47.3818054, -10.8751802, 45.3959198, -56.4806252, 58.2569847
1: -14.2560978, 53.6242714, -13.8271923, 51.3580666, -65.6141663, 67.4514618
2: -13.7948971, 53.5095329, -13.5616131, 51.1347427, -64.9296417, 67.0711441
3: -24.3539143, 56.8776131, -23.3999519, 54.6130333, -78.9669495, 80.2775650
4: -22.2534161, 54.9756432, -21.7295837, 52.5888901, -74.8423080, 76.7052078

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_A1_B2_B1

### Relational analysis result of IS_A2_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.0209083, upper bound: 57.3111963
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_A1_B2_B2

### Relational analysis result of IS_A2_B2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.0553677, upper bound: 57.3156531
time: 0.54 seconds

## BFS IS instance: IS_A2_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -10.6957531, 44.9465866, -10.8751802, 45.3959198, -56.0916672, 55.8217621
1: -13.6549959, 50.8562546, -13.8271923, 51.3580666, -65.0130463, 64.6834488
2: -13.3063459, 50.7025528, -13.5616131, 51.1347427, -64.4410858, 64.2641525
3: -23.1477661, 54.0001144, -23.3999519, 54.6130333, -77.7608032, 77.4000702
4: -21.2595654, 52.1365585, -21.7295837, 52.5888901, -73.8484573, 73.8661270

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_A2_B2_B1

### Relational analysis result of IS_A2_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2091961, upper bound: 57.4665788
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2_B2

### Relational analysis result of IS_A2_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2091961, upper bound: 57.4904413
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -10.0372343, 42.1927071, -11.0297852, 46.1416092, -56.1788368, 53.2224922
1: -12.7809610, 47.7437553, -14.0756989, 52.2037125, -64.9846725, 61.8194542
2: -12.5463152, 47.4627686, -13.7119274, 52.0922241, -64.6385422, 61.1746864
3: -21.6898041, 50.8244820, -23.8188839, 55.4197121, -77.1095123, 74.6433640
4: -20.1472874, 48.8353882, -21.8664837, 53.5777702, -73.7250519, 70.7018738

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4752022, upper bound: 57.4124908
time: 1.01 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4889187, upper bound: 57.4659278
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -10.0193787, 42.8943863, -9.6080275, 40.5105820, -50.5299606, 52.5024147
1: -12.8247375, 48.5298729, -12.2365494, 45.8477631, -58.6725006, 60.7664185
2: -12.5208769, 48.2823448, -12.0228996, 45.5226021, -58.0434799, 60.3052368
3: -21.9548664, 51.6154327, -20.8062382, 48.8360825, -70.7909317, 72.4216690
4: -20.2769165, 49.6436996, -19.3448238, 46.8458099, -67.1227112, 68.9885178

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_B2_A2_B2_A1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2322753, upper bound: 57.3215849
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -57.2045298, upper bound: 57.2045298
time: 1.00 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2045298, upper bound: 57.3737711
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -10.0372343, 42.1927071, -10.4046993, 43.4960098, -53.5332336, 52.5974007
1: -12.7809610, 47.7437553, -13.2431650, 49.2144547, -61.9954071, 60.9869194
2: -12.5463152, 47.4627686, -12.9902172, 48.9800606, -61.5263748, 60.4529800
3: -21.6898041, 50.8244820, -22.4255047, 52.3740425, -74.0638428, 73.2499847
4: -20.1472874, 48.8353882, -20.8130035, 50.4081459, -70.5554123, 69.6483841

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5330534, upper bound: 57.5192887
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5398250, upper bound: 57.5397901
time: 0.76 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.86 seconds
IS_A1_B1_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.4350126, upper bound: 57.4527563
IS_A1_B1_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.4350126, upper bound: 57.4989698
IS_A1_B1_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.4295416, upper bound: 57.4545045
IS_A1_B1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.4295416, upper bound: 57.4972253
IS_A1_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.4859338, upper bound: 57.4494298
IS_A1_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.4859338, upper bound: 57.5061026
IS_A1_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.4859338, upper bound: 57.4771794
IS_A1_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.4859338, upper bound: 57.5061026
IS_A1_B1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.4989698, upper bound: 57.4507852
IS_A1_B1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.4972253, upper bound: 57.4439877
IS_A1_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.5210587, upper bound: 57.5094799
IS_A1_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.5210587, upper bound: 57.5095796
IS_A1_B1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.5092668, upper bound: 57.4692860
IS_A1_B1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.5186089, upper bound: 57.4937896
IS_A1_B1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.5326215, upper bound: 57.5326214
IS_A1_B1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.5326215, upper bound: 57.5326214
IS_A1_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.3971059, upper bound: 57.2644628
IS_A1_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.4044075, upper bound: 57.2659297
IS_A1_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.3564297, upper bound: 57.4670178
IS_A1_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.3564297, upper bound: 57.4835618
IS_A1_B2_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.4226292, upper bound: 57.4776458
IS_A1_B2_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.4226292, upper bound: 57.4939629
IS_A1_B2_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.4551317, upper bound: 57.5093507
IS_A1_B2_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.4551317, upper bound: 57.5199713
IS_A1_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.4227620, upper bound: 57.4464805
IS_A1_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.4227620, upper bound: 57.4696927
IS_A1_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.4969975, upper bound: 57.5155763
IS_A1_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.4961385, upper bound: 57.5115221
IS_A1_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.4630019, upper bound: 57.4096086
IS_A1_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.4630019, upper bound: 57.4986537
IS_A1_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.4732274, upper bound: 57.4617319
IS_A1_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.4732274, upper bound: 57.5315951
IS_A2_B1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.2644628, upper bound: 57.3971059
IS_A2_B1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.2659297, upper bound: 57.4044075
IS_A2_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.2491961, upper bound: 57.4551317
IS_A2_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.2491961, upper bound: 57.5030340
IS_A2_B1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.4776459, upper bound: 57.4352799
IS_A2_B1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.4776459, upper bound: 57.4947398
IS_A2_B1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.5093507, upper bound: 57.4689063
IS_A2_B1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.5093508, upper bound: 57.5168636
IS_A2_B1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.4464805, upper bound: 57.4227620
IS_A2_B1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.4464805, upper bound: 57.4817997
IS_A2_B1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.5155763, upper bound: 57.4969975
IS_A2_B1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.4696927, upper bound: 57.4817997
IS_A2_B1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.4063586, upper bound: 57.4630019
IS_A2_B1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.4063586, upper bound: 57.5259404
IS_A2_B1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.4593124, upper bound: 57.4732274
IS_A2_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.4593124, upper bound: 57.5339194
IS_A2_B2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.0209083, upper bound: 57.3111963
IS_A2_B2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.0553677, upper bound: 57.3156531
IS_A2_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.2091961, upper bound: 57.4665788
IS_A2_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.2091961, upper bound: 57.4904413
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.4752022, upper bound: 57.4124908
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.4889187, upper bound: 57.4659278
IS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.2045298, upper bound: 57.2045298
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.2045298, upper bound: 57.3737711
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.5330534, upper bound: 57.5192887
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.86
Output dim: 0, lower bound: -57.5398250, upper bound: 57.5397901

## BFS IS instance: IS_A1_B1_A1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -5.5928502, 27.0996017, -5.8018961, 26.7496204, -32.3424606, 32.9014931
1: -7.1515408, 30.8633766, -7.3728709, 30.4632874, -37.6148262, 38.2362366
2: -7.1898627, 29.9638920, -7.4531531, 29.5509739, -36.7408333, 37.4170456
3: -12.8735600, 32.9426193, -12.9628153, 32.7265015, -45.6000595, 45.9054298
4: -12.2557392, 30.6781273, -12.5497913, 30.2566566, -42.5123901, 43.2279167

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_A1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4329935, upper bound: 57.4439833
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A1_A1_A1_B1_B2

### Relational analysis result of IS_A1_B1_A1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4329935, upper bound: 57.4527563
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -5.5928502, 27.0996017, -7.1678596, 31.3126755, -36.9055252, 34.2674599
1: -7.1515408, 30.8633766, -9.1584425, 35.5915108, -42.7430458, 40.0218201
2: -7.1898627, 29.9638920, -9.1037121, 34.8285675, -42.0184288, 39.0676003
3: -12.8735600, 32.9426193, -15.7219830, 38.2278175, -51.1013794, 48.6646004
4: -12.2557392, 30.6781273, -14.9093151, 35.8142662, -48.0699959, 45.5874367

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_A1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4329935, upper bound: 57.4919611
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_A1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4329935, upper bound: 57.4989698
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -5.9486384, 27.6570396, -6.5935054, 29.7048607, -35.6534996, 34.2505455
1: -7.5847530, 31.4490318, -8.4099636, 33.7527084, -41.3374596, 39.8589935
2: -7.6243067, 30.6088829, -8.4087238, 32.9591751, -40.5834808, 39.0176010
3: -13.3943548, 33.6805496, -14.6648998, 36.2130852, -49.6074371, 48.3454514
4: -12.7465086, 31.4047012, -13.9864197, 33.8300323, -46.5765343, 45.3911209

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_A1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4250266, upper bound: 57.4395102
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_A1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4250266, upper bound: 57.4475108
time: 0.53 seconds

## BFS IS instance: IS_A1_B1_A1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -5.9486384, 27.6570396, -7.8833742, 33.9009705, -39.8496094, 35.5404129
1: -7.5847530, 31.4490318, -10.0849133, 38.4888535, -46.0736084, 41.5339355
2: -7.6243067, 30.6088829, -9.9639616, 37.8445778, -45.4688835, 40.5728455
3: -13.3943548, 33.6805496, -17.1974564, 41.2503853, -54.6447372, 50.8780060
4: -12.7465086, 31.4047012, -16.1544247, 38.9421730, -51.6886749, 47.5591278

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_A1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4237129, upper bound: 57.4972253
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_A1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4295416, upper bound: 57.4972253
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -7.4350624, 32.4608459, -5.8050623, 26.8103561, -34.2454185, 38.2659073
1: -9.5189266, 36.8422890, -7.4013424, 30.4969997, -40.0159264, 44.2436295
2: -9.4125261, 36.2255974, -7.4495697, 29.6914902, -39.1040154, 43.6751671
3: -16.3478165, 39.4466934, -13.0302229, 32.7092361, -49.0570488, 52.4769096
4: -15.2531500, 37.2985878, -12.4200859, 30.4931049, -45.7462540, 49.7186699

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4145882, upper bound: 57.4155060
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_B2

### Relational analysis result of IS_A1_B1_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4146025, upper bound: 57.4018270
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -7.4350624, 32.4608459, -7.2947378, 31.8497753, -39.2848358, 39.7555771
1: -9.5189266, 36.8422890, -9.3346424, 36.1582756, -45.6772003, 46.1769333
2: -9.4125261, 36.2255974, -9.2406588, 35.5294800, -44.9420052, 45.4662552
3: -16.3478165, 39.4466934, -16.0341167, 38.7422409, -55.0900574, 55.4808006
4: -15.2531500, 37.2985878, -14.9874439, 36.5850677, -51.8382187, 52.2860298

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4167841, upper bound: 57.4687080
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4182548, upper bound: 57.4465134
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -7.4350624, 32.4608459, -5.6721125, 26.4392776, -33.8743362, 38.1329498
1: -9.5189266, 36.8422890, -7.2073269, 30.1333923, -39.6523209, 44.0496140
2: -9.4125261, 36.2255974, -7.3159394, 29.2127800, -38.6253052, 43.5415382
3: -16.3478165, 39.4466934, -12.6886530, 32.3482628, -48.6960793, 52.1353378
4: -15.2531500, 37.2985878, -12.2834435, 29.9198341, -45.1729813, 49.5820312

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_B1

### Relational analysis result of IS_A1_B1_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4865946, upper bound: 57.4648390
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_B2

### Relational analysis result of IS_A1_B1_A1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4940207, upper bound: 57.4771793
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -7.4350624, 32.4608459, -7.3467827, 32.0613708, -39.4964333, 39.8076210
1: -9.5189266, 36.8422890, -9.3942413, 36.4329567, -45.9518814, 46.2365303
2: -9.4125261, 36.2255974, -9.3245058, 35.7049446, -45.1174698, 45.5501022
3: -16.3478165, 39.4466934, -16.1041088, 39.0988007, -55.4466171, 55.5508041
4: -15.2531500, 37.2985878, -15.1873417, 36.7418594, -51.9950104, 52.4859314

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4865946, upper bound: 57.5192981
time: 1.08 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4940208, upper bound: 57.5193461
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -7.9867325, 34.2763634, -5.5928502, 27.0996017, -35.0863304, 39.8692131
1: -10.1965561, 38.8652573, -7.1515408, 30.8633766, -41.0599327, 46.0167961
2: -10.0668869, 38.3097916, -7.1898627, 29.9638920, -40.0307770, 45.4996529
3: -17.4640160, 41.5618553, -12.8735600, 32.9426193, -50.4066353, 54.4354172
4: -16.3867416, 39.4446716, -12.2557392, 30.6781273, -47.0648651, 51.7004051

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B1_B1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4527563, upper bound: 57.4350126
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4527563, upper bound: 57.4507852
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -8.8118200, 37.3577957, -5.9486384, 27.6570396, -36.4688606, 43.3064346
1: -11.2442751, 42.3229713, -7.5847530, 31.4490318, -42.6933060, 49.9077225
2: -11.0679474, 41.8824844, -7.6243067, 30.6088829, -41.6768265, 49.5067902
3: -19.1606712, 45.1869125, -13.3943548, 33.6805496, -52.8412170, 58.5812569
4: -17.8871288, 43.1344337, -12.7465086, 31.4047012, -49.2918282, 55.8809319

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B1_B1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4545045, upper bound: 57.4295416
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4545045, upper bound: 57.4439877
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -6.6899886, 29.3829098, -7.0102844, 30.8641930, -37.5541801, 36.3931961
1: -8.5683136, 33.4162827, -8.9754324, 35.0587959, -43.6271057, 42.3917160
2: -8.5076694, 32.7361832, -8.9031019, 34.3887711, -42.8964386, 41.6392860
3: -14.7535791, 35.8254929, -15.4629374, 37.5761337, -52.3297119, 51.2884254
4: -13.9798346, 33.7439117, -14.4694519, 35.4249268, -49.4047623, 48.2133636

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5197585, upper bound: 57.5075514
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5183954, upper bound: 57.5023501
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -7.3027287, 31.6285458, -7.3282166, 32.0535583, -39.3562851, 38.9567642
1: -9.3334522, 35.9055862, -9.3829374, 36.3860130, -45.7194672, 45.2885208
2: -9.2447405, 35.2926216, -9.2832413, 35.7580643, -45.0028038, 44.5758629
3: -16.0146275, 38.4429092, -16.1267567, 38.9687347, -54.9833603, 54.5696640
4: -15.0447989, 36.3872452, -15.0537300, 36.8225555, -51.8673553, 51.4409676

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1_B1_A2_B1_B2_A2_A1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5095610, upper bound: 57.5031978
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A2_A2

### Relational analysis result of IS_A1_B1_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5184287, upper bound: 57.5079665
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -7.9867325, 34.2763634, -5.1890569, 25.1817741, -33.1685066, 39.4654198
1: -10.1965561, 38.8652573, -6.6473837, 28.7506981, -38.9472504, 45.5126419
2: -10.0668869, 38.3097916, -6.7073054, 27.7961330, -37.8630104, 45.0170937
3: -17.4640160, 41.5618553, -11.9146013, 30.7683716, -48.2323837, 53.4764557
4: -16.3867416, 39.4446716, -11.4802523, 28.4190979, -44.8058319, 50.9249153

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B2_B1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4962781, upper bound: 57.4639298
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A2_B2_B1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4962781, upper bound: 57.4692860
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -8.8118200, 37.3577957, -5.5963645, 26.3081665, -35.1199875, 42.9541588
1: -11.2442751, 42.3229713, -7.1001191, 29.9992542, -41.2435265, 49.4230919
2: -11.0679474, 41.8824844, -7.2258878, 29.0324535, -40.1004028, 49.1083717
3: -19.1606712, 45.1869125, -12.5410995, 32.1875839, -51.3482513, 57.7280121
4: -17.8871288, 43.1344337, -12.1948471, 29.6890259, -47.5761528, 55.3292809

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_B1_B2_B1

### Relational analysis result of IS_A1_B1_A2_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5111073, upper bound: 57.4740232
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B2_B1_B2_B2

### Relational analysis result of IS_A1_B1_A2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5186089, upper bound: 57.4937896
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -7.5570183, 32.6191101, -5.8067570, 26.4507256, -34.0077438, 38.4258652
1: -9.6577930, 37.0172997, -7.4054251, 30.2039413, -39.8617325, 44.4227257
2: -9.5540237, 36.4335632, -7.4816475, 29.2646160, -38.8186302, 43.9152107
3: -16.5514679, 39.6040192, -12.8401928, 32.5236092, -49.0750771, 52.4442139
4: -15.5218801, 37.5543365, -12.4464569, 30.0448952, -45.5667725, 50.0007935

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B2_B2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5168785, upper bound: 57.5119316
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5315038, upper bound: 57.5316335
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -7.8194590, 33.6061020, -6.7064815, 29.6760235, -37.4954834, 40.3125839
1: -9.9893398, 38.1175728, -8.5756721, 33.7746735, -43.7640152, 46.6932449
2: -9.8711519, 37.5690308, -8.5574265, 32.9536819, -42.8248291, 46.1264572
3: -17.0924397, 40.7667961, -14.7597656, 36.2993317, -53.3917694, 55.5265617
4: -16.0185661, 38.7082977, -14.0071154, 33.9241180, -49.9426842, 52.7154121

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_B2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5326215, upper bound: 57.5326214
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5326215, upper bound: 57.5326214
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -8.6716461, 37.7611389, -7.4428377, 33.9592743, -42.6309204, 45.2039757
1: -11.1499243, 42.7709465, -9.5416031, 38.4692078, -49.6191330, 52.3125496
2: -10.8831701, 42.3993225, -9.4428463, 37.8874969, -48.7706680, 51.8421631
3: -19.2832890, 45.5401001, -16.7438488, 41.0775909, -60.3608780, 62.2839508
4: -17.7571411, 43.6625214, -15.7352467, 38.8782501, -56.6353912, 59.3977661

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B1_B1_A1_A1

### Relational analysis result of IS_A1_B2_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3404393, upper bound: 57.2441955
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A1_B1_B1_A1_A2

### Relational analysis result of IS_A1_B2_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3404393, upper bound: 57.2644628
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -8.0646629, 35.2838440, -7.9552212, 35.8122635, -43.8769264, 43.2390633
1: -10.3480911, 39.9815979, -10.2020988, 40.5423241, -50.8904152, 50.1836967
2: -10.1572323, 39.4773331, -10.0640669, 40.0233917, -50.1806259, 49.5414009
3: -17.9082813, 42.6824608, -17.8135185, 43.2676315, -61.1759109, 60.4959793
4: -16.6487427, 40.6530380, -16.6815376, 41.0891838, -57.7379265, 57.3345757

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B1_B1_A2_A1

### Relational analysis result of IS_A1_B2_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3564297, upper bound: 57.2491961
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B1_B1_A2_A2

### Relational analysis result of IS_A1_B2_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3564297, upper bound: 57.2659297
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -6.1764956, 28.8606567, -8.4519434, 37.0708733, -43.2473640, 37.3125954
1: -7.8935847, 32.8192139, -10.7818136, 41.9590607, -49.8526459, 43.6010284
2: -7.9074421, 31.9827385, -10.6651726, 41.4331932, -49.3406334, 42.6479111
3: -14.0097380, 35.1678352, -18.5733147, 44.7520294, -58.7617569, 53.7411499
4: -13.3757744, 32.8016167, -17.3966503, 42.5266571, -55.9024315, 50.1982613

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4226292, upper bound: 57.4346183
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4551317, upper bound: 57.4670178
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -7.8694668, 34.4900322, -8.4519434, 37.0708733, -44.9403381, 42.9419670
1: -10.0883865, 39.1318054, -10.7818136, 41.9590607, -52.0474472, 49.9136200
2: -9.9497414, 38.5054932, -10.6651726, 41.4331932, -51.3829346, 49.1706657
3: -17.4103184, 41.9242973, -18.5733147, 44.7520294, -62.1623383, 60.4976120
4: -16.3477592, 39.6116409, -17.3966503, 42.5266571, -58.8744164, 57.0082855

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4226292, upper bound: 57.4574505
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4551317, upper bound: 57.4835618
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -5.1952600, 25.3797054, -8.4117498, 35.9554672, -41.1507263, 33.7914543
1: -6.6714020, 28.9316158, -10.7277308, 40.7539024, -47.4253044, 39.6593475
2: -6.7203293, 28.0738544, -10.5861588, 40.2136612, -46.9339905, 38.6600113
3: -12.0234785, 30.9182911, -18.2507763, 43.5593109, -55.5827904, 49.1690559
4: -11.4735546, 28.7564545, -17.0741596, 41.3301773, -52.8037338, 45.8306122

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B2_A1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3698477, upper bound: 57.3951798
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3698477, upper bound: 57.4776458
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -6.6024351, 30.2522888, -8.4117498, 35.9554672, -42.5579033, 38.6640396
1: -8.4603643, 34.3885117, -10.7277308, 40.7539024, -49.2142677, 45.1162415
2: -8.4155941, 33.6497993, -10.5861588, 40.2136612, -48.6292534, 44.2359581
3: -14.8398304, 36.7894897, -18.2507763, 43.5593109, -58.3991394, 55.0402603
4: -13.9825315, 34.5861893, -17.0741596, 41.3301773, -55.3127022, 51.6603470

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A1_A2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4224032, upper bound: 57.4903757
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B2_A1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3698477, upper bound: 57.4215482
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3698477, upper bound: 57.4939629
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -5.2425332, 25.4666214, -9.0371084, 38.3006172, -43.5431442, 34.5037231
1: -6.7157836, 29.0644989, -11.5198345, 43.3769875, -50.0927658, 40.5843353
2: -6.7829127, 28.1207294, -11.3451462, 42.9124069, -49.6953163, 39.4658737
3: -12.0501604, 31.1102924, -19.5400734, 46.3099709, -58.3601303, 50.6503677
4: -11.6059141, 28.7633476, -18.2220364, 44.1050758, -55.7109909, 46.9853706

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B2_A2_A1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3732435, upper bound: 57.4363278
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_A1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4551317, upper bound: 57.5047317
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -6.7937622, 30.7798309, -9.0371084, 38.3006172, -45.0943756, 39.8169365
1: -8.6969538, 35.0070839, -11.5198345, 43.3769875, -52.0739403, 46.5269165
2: -8.6619501, 34.1906509, -11.3451462, 42.9124069, -51.5743561, 45.5357933
3: -15.1779556, 37.5357819, -19.5400734, 46.3099709, -61.4879265, 57.0758514
4: -14.3899021, 35.1436996, -18.2220364, 44.1050758, -58.4949799, 53.3657227

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B2_A2_A2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4551085, upper bound: 57.5188253
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_A2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4325800, upper bound: 57.4466343
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -5.9486384, 27.6570396, -8.2398510, 36.1620255, -42.1106644, 35.8968887
1: -7.5847530, 31.4490318, -10.5217819, 40.9423981, -48.5271492, 41.9708099
2: -7.6243067, 30.6088829, -10.4027452, 40.4170303, -48.0413361, 41.0116272
3: -13.3943548, 33.6805496, -18.1303902, 43.6821404, -57.0764923, 51.8109398
4: -12.7465086, 31.4047012, -16.9685364, 41.5019875, -54.2484856, 48.3732338

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_A1_B1_A1_A1

### Relational analysis result of IS_A1_B2_A2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4158336, upper bound: 57.4278544
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_A1_A2

### Relational analysis result of IS_A1_B2_A2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3563228, upper bound: 57.3938980
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -7.0722103, 31.2705612, -8.2398510, 36.1620255, -43.2342377, 39.5104103
1: -9.0512457, 35.5169983, -10.5217819, 40.9423981, -49.9936447, 46.0387764
2: -8.9803925, 34.8309250, -10.4027452, 40.4170303, -49.3974228, 45.2336693
3: -15.6104374, 38.0437126, -18.1303902, 43.6821404, -59.2925797, 56.1741028
4: -14.6354523, 35.8261719, -16.9685364, 41.5019875, -56.1374397, 52.7947083

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_A1_B1_A2_A1

### Relational analysis result of IS_A1_B2_A2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4158336, upper bound: 57.4503942
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_A2_A2

### Relational analysis result of IS_A1_B2_A2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3563228, upper bound: 57.3995690
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -7.3757195, 32.2231102, -7.3329268, 32.0233650, -39.3990860, 39.5560379
1: -9.4408855, 36.5481186, -9.3657732, 36.3913155, -45.8321991, 45.9138870
2: -9.3209267, 35.9883919, -9.2833691, 35.7224808, -45.0434074, 45.2717590
3: -16.2793388, 39.0293846, -16.0269184, 38.9442368, -55.2235756, 55.0563049
4: -15.1506023, 37.1017494, -15.0927572, 36.7110329, -51.8616333, 52.1945076

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4706192, upper bound: 57.4937092
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_B1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4969975, upper bound: 57.5155763
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -7.6697726, 33.3235054, -8.6578579, 36.7560844, -44.4258575, 41.9813538
1: -9.8159800, 37.7809105, -11.0484028, 41.6605682, -51.4765434, 48.8293152
2: -9.6744432, 37.2564621, -10.8807850, 41.1629219, -50.8373642, 48.1372452
3: -16.8923702, 40.3285294, -18.7503777, 44.5058403, -61.3982086, 59.0789032
4: -15.7007055, 38.3920822, -17.4589558, 42.3474655, -58.0481567, 55.8510246

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_A1_B2_B2_B1

### Relational analysis result of IS_A1_B2_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4662952, upper bound: 57.4593124
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_B2_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4662952, upper bound: 57.5115221
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -8.4693804, 36.1834831, -8.0627384, 35.7259407, -44.1953201, 44.2462234
1: -10.8127460, 41.0070839, -10.3429604, 40.4529800, -51.2657242, 51.3500443
2: -10.6570034, 40.5084305, -10.1579933, 39.9914627, -50.6484642, 50.6664238
3: -18.4758568, 43.8035660, -17.8792992, 43.0726242, -61.5484810, 61.6828613
4: -17.2884102, 41.7039909, -16.5315685, 41.0846443, -58.3730545, 58.2355537

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_A2_B1_B1_A1

### Relational analysis result of IS_A1_B2_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4133659, upper bound: 57.3916711
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_B1_A2

### Relational analysis result of IS_A1_B2_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4133659, upper bound: 57.4096086
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -8.4693804, 36.1834831, -7.4656463, 33.3557091, -41.8250847, 43.6491280
1: -10.8127460, 41.0070839, -9.5422745, 37.8093300, -48.6220779, 50.5493546
2: -10.6570034, 40.5084305, -9.4745121, 37.1891479, -47.8461533, 49.9829369
3: -18.4758568, 43.8035660, -16.5416107, 40.3914528, -58.8673096, 60.3451767
4: -17.2884102, 41.7039909, -15.5500689, 38.1762428, -55.4646530, 57.2540588

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_A2_B1_B2_A1

### Relational analysis result of IS_A1_B2_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4133659, upper bound: 57.4807124
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_B2_A2

### Relational analysis result of IS_A1_B2_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4133659, upper bound: 57.4986537
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -7.5869489, 32.8869972, -8.8100138, 37.5939674, -45.1809158, 41.6970100
1: -9.6928911, 37.3132858, -11.2688313, 42.5857544, -52.2786446, 48.5821152
2: -9.5936756, 36.7145805, -11.0355730, 42.1981087, -51.7917862, 47.7501526
3: -16.6362324, 39.9200401, -19.1602993, 45.3824959, -62.0187302, 59.0803375
4: -15.6380339, 37.8018875, -17.6283436, 43.3953476, -59.0333824, 55.4302177

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_A2_B2_B1_A1

### Relational analysis result of IS_A1_B2_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4690091, upper bound: 57.4563847
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_B1_A2

### Relational analysis result of IS_A1_B2_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4732274, upper bound: 57.4617319
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -7.5869489, 32.8869972, -8.7081556, 37.1425133, -44.7294617, 41.5951538
1: -9.6928911, 37.3132858, -11.1119413, 42.0899849, -51.7828751, 48.4252281
2: -9.5936756, 36.7145805, -10.9512491, 41.5937920, -51.1874695, 47.6658287
3: -16.6362324, 39.9200401, -18.8729000, 44.9509163, -61.5871506, 58.7929382
4: -15.6380339, 37.8018875, -17.5746422, 42.7677650, -58.4057999, 55.3765297

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_A1

### Relational analysis result of IS_A1_B2_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4690091, upper bound: 57.5315295
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_B2_A2

### Relational analysis result of IS_A1_B2_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4732274, upper bound: 57.5315950
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -7.4428377, 33.9592743, -8.6716461, 37.7611389, -45.2039757, 42.6309204
1: -9.5416031, 38.4692078, -11.1499243, 42.7709465, -52.3125496, 49.6191330
2: -9.4428463, 37.8874969, -10.8831701, 42.3993225, -51.8421669, 48.7706680
3: -16.7438488, 41.0775909, -19.2832890, 45.5401001, -62.2839508, 60.3608780
4: -15.7352467, 38.8782501, -17.7571411, 43.6625214, -59.3977661, 56.6353912

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_B1_A1_A1_B1_B1

### Relational analysis result of IS_A2_B1_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2441955, upper bound: 57.3404393
time: 0.78 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_B1_B2

### Relational analysis result of IS_A2_B1_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2441955, upper bound: 57.3971059
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -7.9552212, 35.8122635, -8.0646629, 35.2838440, -43.2390671, 43.8769264
1: -10.2020988, 40.5423241, -10.3480911, 39.9815979, -50.1836967, 50.8904152
2: -10.0640669, 40.0233917, -10.1572323, 39.4773331, -49.5414009, 50.1806259
3: -17.8135185, 43.2676315, -17.9082813, 42.6824608, -60.4959793, 61.1759109
4: -16.6815376, 41.0891838, -16.6487427, 40.6530380, -57.3345757, 57.7379265

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_B1_A1_A1_B2_B1

### Relational analysis result of IS_A2_B1_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2491961, upper bound: 57.3564297
time: 0.58 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_B2_B2

### Relational analysis result of IS_A2_B1_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2491961, upper bound: 57.4044075
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -8.4519434, 37.0708733, -6.1764956, 28.8606567, -37.3125954, 43.2473640
1: -10.7818136, 41.9590607, -7.8935847, 32.8192139, -43.6010284, 49.8526459
2: -10.6651726, 41.4331932, -7.9074421, 31.9827385, -42.6479111, 49.3406372
3: -18.5733147, 44.7520294, -14.0097380, 35.1678352, -53.7411461, 58.7617607
4: -17.3966503, 42.5266571, -13.3757744, 32.8016167, -50.1982613, 55.9024315

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B1_A1_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4346183, upper bound: 57.4226292
time: 0.61 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2_B1_B2

### Relational analysis result of IS_A2_B1_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4670178, upper bound: 57.4551317
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -8.4519434, 37.0708733, -7.8694668, 34.4900322, -42.9419670, 44.9403381
1: -10.7818136, 41.9590607, -10.0883865, 39.1318054, -49.9136200, 52.0474472
2: -10.6651726, 41.4331932, -9.9497414, 38.5054932, -49.1706657, 51.3829346
3: -18.5733147, 44.7520294, -17.4103184, 41.9242973, -60.4976120, 62.1623421
4: -17.3966503, 42.5266571, -16.3477592, 39.6116409, -57.0082855, 58.8744125

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B1_A1_A2_B2_B1

### Relational analysis result of IS_A2_B1_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4346183, upper bound: 57.4815300
time: 0.62 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2_B2_B2

### Relational analysis result of IS_A2_B1_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4670178, upper bound: 57.5030340
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -8.4117498, 35.9554672, -5.1952600, 25.3797054, -33.7914543, 41.1507263
1: -10.7277308, 40.7539024, -6.6714020, 28.9316158, -39.6593475, 47.4253044
2: -10.5861588, 40.2136612, -6.7203293, 28.0738544, -38.6600113, 46.9339905
3: -18.2507763, 43.5593109, -12.0234785, 30.9182911, -49.1690559, 55.5827904
4: -17.0741596, 41.3301773, -11.4735546, 28.7564545, -45.8306122, 52.8037338

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B1_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3951798, upper bound: 57.3701333
time: 0.63 seconds

## Relational analysis of IS_A2_B1_B1_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3951798, upper bound: 57.4352799
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -8.4117498, 35.9554672, -6.6024351, 30.2522888, -38.6640396, 42.5579033
1: -10.7277308, 40.7539024, -8.4603643, 34.3885117, -45.1162415, 49.2142677
2: -10.5861588, 40.2136612, -8.4155941, 33.6497993, -44.2359581, 48.6292534
3: -18.2507763, 43.5593109, -14.8398304, 36.7894897, -55.0402603, 58.3991394
4: -17.0741596, 41.3301773, -13.9825315, 34.5861893, -51.6603470, 55.3127022

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_B1_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4663136, upper bound: 57.4843410
time: 0.57 seconds

## Relational analysis of IS_A2_B1_B1_A2_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B1_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3951798, upper bound: 57.4252747
time: 0.59 seconds

## Relational analysis of IS_A2_B1_B1_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3951798, upper bound: 57.4947398
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -9.0371084, 38.3006172, -5.2425332, 25.4666214, -34.5037270, 43.5431442
1: -11.5198345, 43.3769875, -6.7157836, 29.0644989, -40.5843353, 50.0927658
2: -11.3451462, 42.9124069, -6.7829127, 28.1207294, -39.4658737, 49.6953163
3: -19.5400734, 46.3099709, -12.0501604, 31.1102924, -50.6503677, 58.3601303
4: -18.2220364, 44.1050758, -11.6059141, 28.7633476, -46.9853668, 55.7109909

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_B1_A2_B2_B1_B1

### Relational analysis result of IS_A2_B1_B1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4363279, upper bound: 57.3873356
time: 0.66 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_B1_B2

### Relational analysis result of IS_A2_B1_B1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5047317, upper bound: 57.4672196
time: 0.60 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 6.87 seconds
IS_A1_B1_A1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.4329935, upper bound: 57.4439833
IS_A1_B1_A1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.4329935, upper bound: 57.4527563
IS_A1_B1_A1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.4329935, upper bound: 57.4919611
IS_A1_B1_A1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.4329935, upper bound: 57.4989698
IS_A1_B1_A1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.4250266, upper bound: 57.4395102
IS_A1_B1_A1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.4250266, upper bound: 57.4475108
IS_A1_B1_A1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.4237129, upper bound: 57.4972253
IS_A1_B1_A1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.4295416, upper bound: 57.4972253
IS_A1_B1_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.4145882, upper bound: 57.4155060
IS_A1_B1_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.4146025, upper bound: 57.4018270
IS_A1_B1_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.4167841, upper bound: 57.4687080
IS_A1_B1_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.4182548, upper bound: 57.4465134
IS_A1_B1_A1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.4865946, upper bound: 57.4648390
IS_A1_B1_A1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.4940207, upper bound: 57.4771793
IS_A1_B1_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.4865946, upper bound: 57.5192981
IS_A1_B1_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.4940208, upper bound: 57.5193461
IS_A1_B1_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.4527563, upper bound: 57.4350126
IS_A1_B1_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.4527563, upper bound: 57.4507852
IS_A1_B1_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.4545045, upper bound: 57.4295416
IS_A1_B1_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.4545045, upper bound: 57.4439877
IS_A1_B1_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.5197585, upper bound: 57.5075514
IS_A1_B1_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.5183954, upper bound: 57.5023501
IS_A1_B1_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.5095610, upper bound: 57.5031978
IS_A1_B1_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.5184287, upper bound: 57.5079665
IS_A1_B1_A2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.4962781, upper bound: 57.4639298
IS_A1_B1_A2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.4962781, upper bound: 57.4692860
IS_A1_B1_A2_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.5111073, upper bound: 57.4740232
IS_A1_B1_A2_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.5186089, upper bound: 57.4937896
IS_A1_B1_A2_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.5168785, upper bound: 57.5119316
IS_A1_B1_A2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.5315038, upper bound: 57.5316335
IS_A1_B1_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.5326215, upper bound: 57.5326214
IS_A1_B1_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.5326215, upper bound: 57.5326214
IS_A1_B2_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.3404393, upper bound: 57.2441955
IS_A1_B2_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.3404393, upper bound: 57.2644628
IS_A1_B2_A1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.3564297, upper bound: 57.2491961
IS_A1_B2_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.3564297, upper bound: 57.2659297
IS_A1_B2_A1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.4226292, upper bound: 57.4346183
IS_A1_B2_A1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.4551317, upper bound: 57.4670178
IS_A1_B2_A1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.4226292, upper bound: 57.4574505
IS_A1_B2_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.4551317, upper bound: 57.4835618
IS_A1_B2_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.3698477, upper bound: 57.3951798
IS_A1_B2_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.3698477, upper bound: 57.4776458
IS_A1_B2_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.3698477, upper bound: 57.4215482
IS_A1_B2_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.3698477, upper bound: 57.4939629
IS_A1_B2_A1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.3732435, upper bound: 57.4363278
IS_A1_B2_A1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.4551317, upper bound: 57.5047317
IS_A1_B2_A1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.4551085, upper bound: 57.5188253
IS_A1_B2_A1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.4325800, upper bound: 57.4466343
IS_A1_B2_A2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.4158336, upper bound: 57.4278544
IS_A1_B2_A2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.3563228, upper bound: 57.3938980
IS_A1_B2_A2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.4158336, upper bound: 57.4503942
IS_A1_B2_A2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.3563228, upper bound: 57.3995690
IS_A1_B2_A2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.4706192, upper bound: 57.4937092
IS_A1_B2_A2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.4969975, upper bound: 57.5155763
IS_A1_B2_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.4662952, upper bound: 57.4593124
IS_A1_B2_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.4662952, upper bound: 57.5115221
IS_A1_B2_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.4133659, upper bound: 57.3916711
IS_A1_B2_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.4133659, upper bound: 57.4096086
IS_A1_B2_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.4133659, upper bound: 57.4807124
IS_A1_B2_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.4133659, upper bound: 57.4986537
IS_A1_B2_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.4690091, upper bound: 57.4563847
IS_A1_B2_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.4732274, upper bound: 57.4617319
IS_A1_B2_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.4690091, upper bound: 57.5315295
IS_A1_B2_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.4732274, upper bound: 57.5315950
IS_A2_B1_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.2441955, upper bound: 57.3404393
IS_A2_B1_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.2441955, upper bound: 57.3971059
IS_A2_B1_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.2491961, upper bound: 57.3564297
IS_A2_B1_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.2491961, upper bound: 57.4044075
IS_A2_B1_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.4346183, upper bound: 57.4226292
IS_A2_B1_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.4670178, upper bound: 57.4551317
IS_A2_B1_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.4346183, upper bound: 57.4815300
IS_A2_B1_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.4670178, upper bound: 57.5030340
IS_A2_B1_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.3951798, upper bound: 57.3701333
IS_A2_B1_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.3951798, upper bound: 57.4352799
IS_A2_B1_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.3951798, upper bound: 57.4252747
IS_A2_B1_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.3951798, upper bound: 57.4947398
IS_A2_B1_B1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.4363279, upper bound: 57.3873356
IS_A2_B1_B1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 6.87
Output dim: 0, lower bound: -57.5047317, upper bound: 57.4672196
IS_A2_B1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.87
Output dim: 0, lower bound: -57.5093508, upper bound: 57.5168636
IS_A2_B1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.87
Output dim: 0, lower bound: -57.4464805, upper bound: 57.4227620
IS_A2_B1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.87
Output dim: 0, lower bound: -57.4464805, upper bound: 57.4817997
IS_A2_B1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.87
Output dim: 0, lower bound: -57.5155763, upper bound: 57.4969975
IS_A2_B1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.87
Output dim: 0, lower bound: -57.4696927, upper bound: 57.4817997
IS_A2_B1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 6.87
Output dim: 0, lower bound: -57.4063586, upper bound: 57.4630019
IS_A2_B1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 6.87
Output dim: 0, lower bound: -57.4063586, upper bound: 57.5259404
IS_A2_B1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.87
Output dim: 0, lower bound: -57.4593124, upper bound: 57.4732274
IS_A2_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.87
Output dim: 0, lower bound: -57.4593124, upper bound: 57.5339194
IS_A2_B2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.87
Output dim: 0, lower bound: -57.0209083, upper bound: 57.3111963
IS_A2_B2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.87
Output dim: 0, lower bound: -57.0553677, upper bound: 57.3156531
IS_A2_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.87
Output dim: 0, lower bound: -57.2091961, upper bound: 57.4665788
IS_A2_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.87
Output dim: 0, lower bound: -57.2091961, upper bound: 57.4904413
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.87
Output dim: 0, lower bound: -57.4752022, upper bound: 57.4124908
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.87
Output dim: 0, lower bound: -57.4889187, upper bound: 57.4659278
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.87
Output dim: 0, lower bound: -57.2045298, upper bound: 57.3737711
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.87
Output dim: 0, lower bound: -57.5330534, upper bound: 57.5192887
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.87
Output dim: 0, lower bound: -57.5398250, upper bound: 57.5397901
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=66.57380676269531
rel_dist={0: [-57.5687467976788, 57.5687467976788]}

## Binary search (step 2) starts
Candidate diff: 0.0625000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5526292, upper bound: 57.5552693
time: 0.53 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5620108, upper bound: 57.5620108
time: 0.57 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.29 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.29
Output dim: 0, lower bound: -57.5526292, upper bound: 57.5552693
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.29
Output dim: 0, lower bound: -57.5620108, upper bound: 57.5620108

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -10.1770201, 42.4380188, -11.4069080, 47.1075592, -57.2845764, 53.8449249
1: -12.9492474, 48.0443954, -14.4834604, 53.3023415, -66.2515869, 62.5278549
2: -12.7204800, 47.7395439, -14.2116318, 53.1170883, -65.8375702, 61.9511719
3: -21.9290562, 51.2056084, -24.4205017, 56.6857185, -78.6147614, 75.6261063
4: -20.4464874, 49.1660233, -22.7181416, 54.7416573, -75.1881332, 71.8841629

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4934149, upper bound: 57.5019506
time: 0.54 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5526292, upper bound: 57.5552693
time: 0.53 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -11.9793425, 49.4708633, -12.7021961, 52.2306023, -64.2099457, 62.1730576
1: -15.2106190, 55.9559326, -16.1179504, 59.0786743, -74.2892914, 72.0738754
2: -14.9106045, 55.8369446, -15.7900686, 59.0102730, -73.9208755, 71.6270142
3: -25.6317139, 59.4555740, -27.1216049, 62.7400017, -88.3717117, 86.5771790
4: -23.7928352, 57.5261345, -25.1618710, 60.8016586, -84.5944977, 82.6879959

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4753677, upper bound: 57.4856191
time: 0.56 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5620108, upper bound: 57.5620108
time: 0.64 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.34 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 3.34
Output dim: 0, lower bound: -57.4934149, upper bound: 57.5019506
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 3.34
Output dim: 0, lower bound: -57.5526292, upper bound: 57.5552693
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 3.34
Output dim: 0, lower bound: -57.4753677, upper bound: 57.4856191
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 3.34
Output dim: 0, lower bound: -57.5620108, upper bound: 57.5620108

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -9.1575928, 38.8744278, -10.1699133, 42.4297638, -51.5873566, 49.0443382
1: -11.6959438, 44.0212021, -12.9333305, 48.0276260, -59.7235718, 56.9545326
2: -11.4683819, 43.6604576, -12.6975451, 47.7412872, -59.2096672, 56.3580017
3: -19.9659348, 46.8748016, -21.8981991, 51.1333847, -71.0993042, 68.7730026
4: -18.4506874, 44.9865723, -20.3782845, 49.1402321, -67.5909195, 65.3648529

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4934149, upper bound: 57.5019506
time: 0.58 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4934149, upper bound: 57.5019506
time: 0.60 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -8.8118200, 37.3577957, -10.8023367, 44.8221817, -53.6340027, 48.1601257
1: -11.2442751, 42.3229713, -13.7275448, 50.7243805, -61.9686546, 56.0505142
2: -11.0679474, 41.8824844, -13.4740314, 50.4916763, -61.5596237, 55.3565140
3: -19.1606712, 45.1869125, -23.1908646, 53.9760590, -73.1367340, 68.3777695
4: -17.8871288, 43.1344337, -21.5745335, 52.0141296, -69.9012451, 64.7089691

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5512705, upper bound: 57.5512705
time: 0.54 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5512705, upper bound: 57.5552693
time: 0.55 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -11.0297852, 46.1416092, -11.4109707, 47.3582001, -58.3879852, 57.5525780
1: -14.0756989, 52.2037125, -14.5050850, 53.5862274, -67.6619263, 66.7088013
2: -13.7119274, 52.0922241, -14.2091827, 53.4185448, -67.1304550, 66.3014069
3: -23.8188839, 55.4197121, -24.4906864, 56.9475174, -80.7664032, 79.9104004
4: -21.8664837, 53.5777702, -22.7076092, 54.9755669, -76.8420486, 76.2853775

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4728293, upper bound: 57.4790544
time: 0.56 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4728293, upper bound: 57.4790544
time: 0.53 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -10.4046993, 43.4960098, -12.0870428, 49.9106674, -60.3153687, 55.5830460
1: -13.2431650, 49.2144547, -15.3528671, 56.4631310, -69.7062988, 64.5673218
2: -12.9902172, 48.9800606, -15.0387974, 56.3477173, -69.3379135, 64.0188599
3: -22.4255047, 52.3740425, -25.8686085, 59.9844704, -82.4099731, 78.2426453
4: -20.8130035, 50.4081459, -23.9925995, 58.0296860, -78.8426895, 74.4007263

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5552692, upper bound: 57.5526292
time: 0.59 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5552693, upper bound: 57.5619695
time: 0.58 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.34 seconds
IS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 3.34
Output dim: 0, lower bound: -57.4934149, upper bound: 57.5019506
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.34
Output dim: 0, lower bound: -57.4934149, upper bound: 57.5019506
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 3.34
Output dim: 0, lower bound: -57.5512705, upper bound: 57.5512705
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 3.34
Output dim: 0, lower bound: -57.5512705, upper bound: 57.5552693
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 3.34
Output dim: 0, lower bound: -57.4728293, upper bound: 57.4790544
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.34
Output dim: 0, lower bound: -57.4728293, upper bound: 57.4790544
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 3.34
Output dim: 0, lower bound: -57.5552692, upper bound: 57.5526292
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 3.34
Output dim: 0, lower bound: -57.5552693, upper bound: 57.5619695

## BFS IS instance: IS_A1_A1_B1

### Backsubstitution after applying IS history:
0: -9.1575928, 38.8744278, -9.0086327, 38.0099335, -47.1675224, 47.8830490
1: -11.6959438, 44.0212021, -11.4879093, 43.0482635, -54.7442093, 55.5091095
2: -11.4683819, 43.6604576, -11.2916431, 42.6426582, -54.1110382, 54.9520988
3: -19.9659348, 46.8748016, -19.5446796, 45.9323807, -65.8983154, 66.4194794
4: -18.4506874, 44.9865723, -18.2404270, 43.8986626, -62.3493500, 63.2269974

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_A1_B1_A1

### Relational analysis result of IS_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4154537, upper bound: 57.4637674
time: 0.59 seconds

## Relational analysis of IS_A1_A1_B1_A2

### Relational analysis result of IS_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4769667, upper bound: 57.4865724
time: 0.64 seconds

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: -9.1575928, 38.8744278, -10.6690159, 44.5334511, -53.6910400, 49.5434303
1: -11.6959438, 44.0212021, -13.5670176, 50.3871536, -62.0830994, 57.5882187
2: -11.4683819, 43.6604576, -13.3105011, 50.1640015, -61.6323853, 56.9709587
3: -19.9659348, 46.8748016, -22.9631214, 53.5931091, -73.5590363, 69.8379211
4: -18.4506874, 44.9865723, -21.3175278, 51.6304970, -70.0811768, 66.3041000

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4664976, upper bound: 57.4390161
time: 0.59 seconds

## Relational analysis of IS_A1_A1_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4769667, upper bound: 57.4865724
time: 0.63 seconds

## BFS IS instance: IS_A1_A2_B1

### Backsubstitution after applying IS history:
0: -8.8118200, 37.3577957, -9.5359230, 40.0448875, -48.8567085, 46.8937187
1: -11.2442751, 42.3229713, -12.1480436, 45.3438530, -56.5881271, 54.4710121
2: -11.0679474, 41.8824844, -11.9415979, 44.9866333, -56.0545769, 53.8240814
3: -19.1606712, 45.1869125, -20.6249313, 48.3589439, -67.5196075, 65.8118286
4: -17.8871288, 43.1344337, -19.2402859, 46.3154488, -64.2025604, 62.3747177

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_A2_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5195957, upper bound: 57.4942538
time: 0.54 seconds

## Relational analysis of IS_A1_A2_B1_B2

### Relational analysis result of IS_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5326215, upper bound: 57.5326214
time: 0.63 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: -8.8118200, 37.3577957, -11.3693390, 47.1662560, -55.9780769, 48.7271347
1: -11.2442751, 42.3229713, -14.4490643, 53.3557320, -64.6000061, 56.7720337
2: -11.0679474, 41.8824844, -14.1657600, 53.1915741, -64.2595215, 56.0482445
3: -19.1606712, 45.1869125, -24.3913670, 56.7219086, -75.8825836, 69.5782700
4: -17.8871288, 43.1344337, -22.6368389, 54.7775879, -72.6647034, 65.7712708

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_A2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5195957, upper bound: 57.5078869
time: 0.82 seconds

## Relational analysis of IS_A1_A2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5326215, upper bound: 57.5367500
time: 0.81 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: -11.0297852, 46.1416092, -9.0654306, 38.2294083, -49.2591934, 55.2070389
1: -14.0756989, 52.2037125, -11.5589437, 43.2975159, -57.3732147, 63.7626572
2: -13.7119274, 52.0922241, -11.3617239, 42.8905373, -56.6024590, 63.4539490
3: -23.8188839, 55.4197121, -19.6654778, 46.1984787, -70.0173645, 75.0851898
4: -21.8664837, 53.5777702, -18.3505554, 44.1525192, -66.0190048, 71.9282990

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_A1_B1_A1

### Relational analysis result of IS_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4047354, upper bound: 57.4525981
time: 0.57 seconds

## Relational analysis of IS_A2_A1_B1_A2

### Relational analysis result of IS_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4564568, upper bound: 57.4623107
time: 0.61 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: -11.0297852, 46.1416092, -10.7050171, 44.6627159, -55.6925011, 56.8466263
1: -14.0756989, 52.2037125, -13.6141796, 50.5332298, -64.6089325, 65.8178940
2: -13.7119274, 52.0922241, -13.3514185, 50.3138771, -64.0258026, 65.4436417
3: -23.8188839, 55.4197121, -23.0363426, 53.7460365, -77.5649185, 78.4560547
4: -21.8664837, 53.5777702, -21.3794994, 51.7781792, -73.6446609, 74.9572678

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_A1_B2_A1

### Relational analysis result of IS_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4047354, upper bound: 57.4525981
time: 0.56 seconds

## Relational analysis of IS_A2_A1_B2_A2

### Relational analysis result of IS_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4564568, upper bound: 57.4623107
time: 0.59 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -10.4046993, 43.4960098, -9.6058359, 40.3157463, -50.7204399, 53.1018333
1: -13.2431650, 49.2144547, -12.2354774, 45.6513290, -58.8944893, 61.4499321
2: -12.9902172, 48.9800606, -12.0278358, 45.2923660, -58.2825851, 61.0078964
3: -22.4255047, 52.3740425, -20.7736835, 48.6870728, -71.1125717, 73.1477203
4: -20.8130035, 50.4081459, -19.3757629, 46.6283188, -67.4413223, 69.7838898

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4856191, upper bound: 57.4753676
time: 0.59 seconds

## Relational analysis of IS_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4668972, upper bound: 57.5526292
time: 0.66 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -10.4046993, 43.4960098, -11.3788996, 47.2034073, -57.6081009, 54.8749084
1: -13.2431650, 49.2144547, -14.4609737, 53.3978958, -66.6410599, 63.6754303
2: -12.9902172, 48.9800606, -14.1775379, 53.2335358, -66.2237473, 63.1576004
3: -22.4255047, 52.3740425, -24.4117718, 56.7669830, -79.1924896, 76.7858124
4: -20.8130035, 50.4081459, -22.6555748, 54.8203659, -75.6333618, 73.0636902

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5228695, upper bound: 57.5192622
time: 0.59 seconds

## Relational analysis of IS_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5367501, upper bound: 57.5396335
time: 0.59 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.37 seconds
IS_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 0, lower bound: -57.4154537, upper bound: 57.4637674
IS_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 0, lower bound: -57.4769667, upper bound: 57.4865724
IS_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 0, lower bound: -57.4664976, upper bound: 57.4390161
IS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 0, lower bound: -57.4769667, upper bound: 57.4865724
IS_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 0, lower bound: -57.5195957, upper bound: 57.4942538
IS_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 0, lower bound: -57.5326215, upper bound: 57.5326214
IS_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 0, lower bound: -57.5195957, upper bound: 57.5078869
IS_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 0, lower bound: -57.5326215, upper bound: 57.5367500
IS_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 0, lower bound: -57.4047354, upper bound: 57.4525981
IS_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 0, lower bound: -57.4564568, upper bound: 57.4623107
IS_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 0, lower bound: -57.4047354, upper bound: 57.4525981
IS_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 0, lower bound: -57.4564568, upper bound: 57.4623107
IS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 0, lower bound: -57.4856191, upper bound: 57.4753676
IS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 0, lower bound: -57.4668972, upper bound: 57.5526292
IS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 0, lower bound: -57.5228695, upper bound: 57.5192622
IS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 0, lower bound: -57.5367501, upper bound: 57.5396335

## BFS IS instance: IS_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -6.2563448, 28.6933155, -9.0086327, 38.0099335, -44.2662735, 37.7019424
1: -7.9915113, 32.5977592, -11.4879093, 43.0482635, -51.0397720, 44.0856590
2: -7.9890079, 31.8166008, -11.2916431, 42.6426582, -50.6316605, 43.1082382
3: -14.0460072, 34.9063377, -19.5446796, 45.9323807, -59.9783859, 54.4510193
4: -13.2832928, 32.6852341, -18.2404270, 43.8986626, -57.1819458, 50.9256592

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_A1_B1_A1_B1

### Relational analysis result of IS_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4333312, upper bound: 57.4439078
time: 0.52 seconds

## Relational analysis of IS_A1_A1_B1_A1_B2

### Relational analysis result of IS_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4333312, upper bound: 57.4934621
time: 0.54 seconds

## BFS IS instance: IS_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -7.4350624, 32.4608459, -7.5147343, 32.3412209, -39.7762833, 39.9755783
1: -9.5189266, 36.8422890, -9.5979338, 36.6917114, -46.2106400, 46.4402237
2: -9.4125261, 36.2255974, -9.4864607, 36.1122971, -45.5248222, 45.7120590
3: -16.3478165, 39.4466934, -16.4404678, 39.2663498, -55.6141663, 55.8871613
4: -15.2531500, 37.2985878, -15.4489317, 37.1850853, -52.4382362, 52.7475204

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_A1_B1_A2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5061430, upper bound: 57.5061431
time: 0.58 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5061430, upper bound: 57.5061431
time: 0.58 seconds

## BFS IS instance: IS_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -9.1575928, 38.8744278, -7.7548656, 34.3289261, -43.4865189, 46.6292915
1: -11.6959438, 44.0212021, -9.9109211, 38.8927612, -50.5887070, 53.9321213
2: -11.4683819, 43.6604576, -9.8113804, 38.3144073, -49.7827911, 53.4718323
3: -19.9659348, 46.8748016, -17.1326771, 41.5241203, -61.4900436, 64.0074692
4: -18.4506874, 44.9865723, -16.0652657, 39.3405418, -57.7912292, 61.0518379

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_A1_B2_B1_A1

### Relational analysis result of IS_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4059562, upper bound: 57.4155201
time: 0.61 seconds

## Relational analysis of IS_A1_A1_B2_B1_A2

### Relational analysis result of IS_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4059562, upper bound: 57.4390161
time: 0.57 seconds

## BFS IS instance: IS_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -7.4272614, 32.2916946, -8.6534843, 36.8002853, -44.2275467, 40.9451790
1: -9.5002413, 36.6233063, -11.0422096, 41.6996994, -51.1999359, 47.6655159
2: -9.3774567, 36.0825157, -10.8732853, 41.2166405, -50.5940933, 46.9557953
3: -16.3603516, 39.1245842, -18.7433796, 44.5373421, -60.8976936, 57.8679504
4: -15.2274914, 37.1916122, -17.4636860, 42.3764076, -57.6038971, 54.6552963

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_A1_B2_B2_B1

### Relational analysis result of IS_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4587161, upper bound: 57.4554520
time: 0.56 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4587161, upper bound: 57.4865724
time: 0.65 seconds

## BFS IS instance: IS_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -8.8118200, 37.3577957, -6.5372963, 29.5482712, -38.3600922, 43.8950920
1: -11.2442751, 42.3229713, -8.3372393, 33.5832443, -44.8275185, 50.6602020
2: -11.0679474, 41.8824844, -8.3474846, 32.7798882, -43.8478317, 50.2299690
3: -19.1606712, 45.1869125, -14.5443096, 36.0342140, -55.1948776, 59.7312164
4: -17.8871288, 43.1344337, -13.8917007, 33.6436195, -51.5307465, 57.0261269

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_A2_B1_B1_A1

### Relational analysis result of IS_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4812687, upper bound: 57.4812687
time: 0.56 seconds

## Relational analysis of IS_A1_A2_B1_B1_A2

### Relational analysis result of IS_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4812687, upper bound: 57.4942538
time: 0.56 seconds

## BFS IS instance: IS_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -7.3673935, 31.9134426, -7.8683906, 33.8547401, -41.2221260, 39.7818260
1: -9.4090042, 36.2246475, -10.0630388, 38.4357719, -47.8447723, 46.2876854
2: -9.3240471, 35.6100235, -9.9473591, 37.7975845, -47.1216278, 45.5573807
3: -16.1484604, 38.7872200, -17.1524582, 41.1903610, -57.3388214, 55.9396782
4: -15.1940689, 36.6823387, -16.1190357, 38.8968315, -54.0908966, 52.8013763

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A2_B1_B2_B1

### Relational analysis result of IS_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5326215, upper bound: 57.5326214
time: 0.68 seconds

## Relational analysis of IS_A1_A2_B1_B2_B2

### Relational analysis result of IS_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5326215, upper bound: 57.5326214
time: 0.81 seconds

## BFS IS instance: IS_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -8.8118200, 37.3577957, -8.2839651, 36.3554649, -45.1672859, 45.6417618
1: -11.2442751, 42.3229713, -10.5785780, 41.1595268, -52.4038010, 52.9015503
2: -11.0679474, 41.8824844, -10.4618778, 40.6369743, -51.7049217, 52.3443604
3: -19.1606712, 45.1869125, -18.2245331, 43.9165955, -63.0772667, 63.4114265
4: -17.8871288, 43.1344337, -17.0619030, 41.7284966, -59.6156235, 60.1963348

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_A2_B2_B1_A1

### Relational analysis result of IS_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4845372, upper bound: 57.4918280
time: 0.58 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2

### Relational analysis result of IS_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4845372, upper bound: 57.5078869
time: 0.56 seconds

## BFS IS instance: IS_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -7.3673935, 31.9134426, -9.3014174, 39.2170753, -46.5844574, 41.2148590
1: -9.4090042, 36.2246475, -11.8640585, 44.4042320, -53.8132362, 48.0887032
2: -9.3240471, 35.6100235, -11.6631804, 44.0017433, -53.3257866, 47.2732048
3: -16.1484604, 38.7872200, -20.0760365, 47.3849373, -63.5333939, 58.8632545
4: -15.1940689, 36.6823387, -18.6597080, 45.2499275, -60.4439964, 55.3420486

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A2_B2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5335269, upper bound: 57.5367441
time: 0.84 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5331228, upper bound: 57.5353124
time: 0.57 seconds

## BFS IS instance: IS_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -8.0627384, 35.7259407, -9.0654306, 38.2294083, -46.2921448, 44.7913704
1: -10.3429604, 40.4529800, -11.5589437, 43.2975159, -53.6404762, 52.0119209
2: -10.1579933, 39.9914627, -11.3617239, 42.8905373, -53.0485268, 51.3531837
3: -17.8792992, 43.0726242, -19.6654778, 46.1984787, -64.0777740, 62.7381020
4: -16.5315685, 41.0846443, -18.3505554, 44.1525192, -60.6840820, 59.4351997

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_A1_B1_A1_B1

### Relational analysis result of IS_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3851447, upper bound: 57.3966763
time: 0.58 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2

### Relational analysis result of IS_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3851447, upper bound: 57.4525981
time: 0.57 seconds

## BFS IS instance: IS_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -9.0433788, 38.5837364, -7.5437603, 32.4550476, -41.4984283, 46.1274910
1: -11.5689020, 43.6965904, -9.6343040, 36.8212395, -48.3901405, 53.3308945
2: -11.3229685, 43.3287659, -9.5224276, 36.2404327, -47.5634003, 52.8511925
3: -19.6716652, 46.5399094, -16.5025177, 39.4043388, -59.0760040, 63.0424271
4: -18.0751343, 44.5710564, -15.5054188, 37.3169060, -55.3920403, 60.0764771

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_A1_B1_A2_B1

### Relational analysis result of IS_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4554520, upper bound: 57.4587161
time: 0.56 seconds

## Relational analysis of IS_A2_A1_B1_A2_B2

### Relational analysis result of IS_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4554520, upper bound: 57.4623107
time: 0.68 seconds

## BFS IS instance: IS_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -8.0627384, 35.7259407, -10.7050171, 44.6627159, -52.7254562, 46.4309578
1: -10.3429604, 40.4529800, -13.6141796, 50.5332298, -60.8761902, 54.0671539
2: -10.1579933, 39.9914627, -13.3514185, 50.3138771, -60.4718704, 53.3428802
3: -17.8792992, 43.0726242, -23.0363426, 53.7460365, -71.6253357, 66.1089630
4: -16.5315685, 41.0846443, -21.3794994, 51.7781792, -68.3097382, 62.4641418

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_A1_B2_A1_B1

### Relational analysis result of IS_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3851447, upper bound: 57.3966763
time: 0.56 seconds

## Relational analysis of IS_A2_A1_B2_A1_B2

### Relational analysis result of IS_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3851447, upper bound: 57.4525981
time: 0.57 seconds

## BFS IS instance: IS_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -9.0433788, 38.5837364, -8.7460098, 37.1486549, -46.1920319, 47.3297462
1: -11.5689020, 43.6965904, -11.1394768, 42.0526276, -53.6215286, 54.8360672
2: -11.3229685, 43.3287659, -10.9823704, 41.6653748, -52.9883423, 54.3111343
3: -19.6716652, 46.5399094, -18.9741840, 44.8549118, -64.5265808, 65.5140915
4: -18.0751343, 44.5710564, -17.6813164, 42.8905334, -60.9656677, 62.2523689

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 26

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_A1_B2_A2_B1

### Relational analysis result of IS_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4354016, upper bound: 57.4008421
time: 0.55 seconds

## Relational analysis of IS_A2_A1_B2_A2_B2

### Relational analysis result of IS_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4354016, upper bound: 57.4623107
time: 0.98 seconds

## BFS IS instance: IS_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -10.4046993, 43.4960098, -9.1575928, 38.8744278, -49.2791176, 52.6535950
1: -13.2431650, 49.2144547, -11.6959438, 44.0212021, -57.2643661, 60.9104004
2: -12.9902172, 48.9800606, -11.4683819, 43.6604576, -56.6506729, 60.4484406
3: -22.4255047, 52.3740425, -19.9659348, 46.8748016, -69.3003006, 72.3399658
4: -20.8130035, 50.4081459, -18.4506874, 44.9865723, -65.7995758, 68.8588257

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_A2_B1_B1_A1

### Relational analysis result of IS_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4390161, upper bound: 57.4664976
time: 0.58 seconds

## Relational analysis of IS_A2_A2_B1_B1_A2

### Relational analysis result of IS_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4865724, upper bound: 57.4769667
time: 0.61 seconds

## BFS IS instance: IS_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -10.4046993, 43.4960098, -8.8118200, 37.3577957, -47.7624893, 52.3078308
1: -13.2431650, 49.2144547, -11.2442751, 42.3229713, -55.5661354, 60.4587288
2: -12.9902172, 48.9800606, -11.0679474, 41.8824844, -54.8726997, 60.0480080
3: -22.4255047, 52.3740425, -19.1606712, 45.1869125, -67.6124115, 71.5347061
4: -20.8130035, 50.4081459, -17.8871288, 43.1344337, -63.9474335, 68.2952499

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_A2_B1_B2_A1

### Relational analysis result of IS_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4390161, upper bound: 57.5238990
time: 0.62 seconds

## Relational analysis of IS_A2_A2_B1_B2_A2

### Relational analysis result of IS_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4865724, upper bound: 57.5337611
time: 0.65 seconds

## BFS IS instance: IS_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -10.4046993, 43.4960098, -8.2914886, 36.3861198, -46.7908134, 51.7874908
1: -13.2431650, 49.2144547, -10.5880194, 41.1941528, -54.4373169, 59.8024750
2: -12.9902172, 48.9800606, -10.4712791, 40.6714973, -53.6617126, 59.4513397
3: -22.4255047, 52.3740425, -18.2408638, 43.9536514, -66.3791580, 70.6148987
4: -20.8130035, 50.4081459, -17.0769653, 41.7640762, -62.5770760, 67.4851074

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_A2_B2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5126623, upper bound: 57.5121573
time: 0.58 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5126623, upper bound: 57.5192622
time: 0.66 seconds

## BFS IS instance: IS_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -8.6222887, 36.7295456, -9.4038916, 39.6162376, -48.2385254, 46.1334381
1: -10.9891005, 41.5826263, -11.9946785, 44.8572311, -55.8463326, 53.5773048
2: -10.8409109, 41.1832390, -11.7895679, 44.4554787, -55.2963905, 52.9728088
3: -18.7244911, 44.3607635, -20.2966499, 47.8671532, -66.5916443, 64.6574020
4: -17.4421425, 42.4057846, -18.8562984, 45.7171288, -63.1592674, 61.2620850

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_A2_B2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4714207, upper bound: 57.4591648
time: 0.59 seconds

## Relational analysis of IS_A2_A2_B2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4714207, upper bound: 57.5396309
time: 0.58 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.88 seconds
IS_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -57.4333312, upper bound: 57.4439078
IS_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -57.4333312, upper bound: 57.4934621
IS_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -57.5061430, upper bound: 57.5061431
IS_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -57.5061430, upper bound: 57.5061431
IS_A1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -57.4059562, upper bound: 57.4155201
IS_A1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -57.4059562, upper bound: 57.4390161
IS_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -57.4587161, upper bound: 57.4554520
IS_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -57.4587161, upper bound: 57.4865724
IS_A1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -57.4812687, upper bound: 57.4812687
IS_A1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -57.4812687, upper bound: 57.4942538
IS_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -57.5326215, upper bound: 57.5326214
IS_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -57.5326215, upper bound: 57.5326214
IS_A1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -57.4845372, upper bound: 57.4918280
IS_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -57.4845372, upper bound: 57.5078869
IS_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -57.5335269, upper bound: 57.5367441
IS_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -57.5331228, upper bound: 57.5353124
IS_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -57.3851447, upper bound: 57.3966763
IS_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -57.3851447, upper bound: 57.4525981
IS_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -57.4554520, upper bound: 57.4587161
IS_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -57.4554520, upper bound: 57.4623107
IS_A2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -57.3851447, upper bound: 57.3966763
IS_A2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -57.3851447, upper bound: 57.4525981
IS_A2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -57.4354016, upper bound: 57.4008421
IS_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -57.4354016, upper bound: 57.4623107
IS_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -57.4390161, upper bound: 57.4664976
IS_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -57.4865724, upper bound: 57.4769667
IS_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -57.4390161, upper bound: 57.5238990
IS_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -57.4865724, upper bound: 57.5337611
IS_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -57.5126623, upper bound: 57.5121573
IS_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -57.5126623, upper bound: 57.5192622
IS_A2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -57.4714207, upper bound: 57.4591648
IS_A2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -57.4714207, upper bound: 57.5396309

## BFS IS instance: IS_A1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -6.2563448, 28.6933155, -6.1247592, 27.9203396, -34.1766853, 34.8180733
1: -7.9915113, 32.5977592, -7.8091989, 31.7582989, -39.7498055, 40.4069557
2: -7.9890079, 31.8166008, -7.8384829, 30.9134712, -38.9024811, 39.6550751
3: -14.0460072, 34.9063377, -13.6689997, 34.0950699, -48.1410751, 48.5753365
4: -13.2832928, 32.6852341, -13.1040649, 31.7116013, -44.9948921, 45.7892990

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4292205, upper bound: 57.4292205
time: 0.56 seconds

## Relational analysis of IS_A1_A1_B1_A1_B1_B2

### Relational analysis result of IS_A1_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4292205, upper bound: 57.4439078
time: 0.60 seconds

## BFS IS instance: IS_A1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6.2563448, 28.6933155, -7.3145170, 31.7948856, -38.0512314, 36.0078316
1: -7.9915113, 32.5977592, -9.3592911, 36.1214294, -44.1129417, 41.9570427
2: -7.9890079, 31.8166008, -9.2725973, 35.4209518, -43.4099541, 41.0891991
3: -14.0460072, 34.9063377, -16.0183315, 38.7490196, -52.7950287, 50.9246674
4: -13.2832928, 32.6852341, -15.1029406, 36.4365845, -49.7198715, 47.7881737

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3731779, upper bound: 57.4773069
time: 0.59 seconds

## Relational analysis of IS_A1_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4333312, upper bound: 57.4934621
time: 0.73 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -7.4350624, 32.4608459, -7.2635403, 31.5559902, -38.9910507, 39.7243805
1: -9.5189266, 36.8422890, -9.2872677, 35.7980232, -45.3169479, 46.1295547
2: -9.4125261, 36.2255974, -9.1751041, 35.2483826, -44.6609077, 45.4007034
3: -16.3478165, 39.4466934, -15.9887466, 38.2698936, -54.6177101, 55.4354401
4: -15.2531500, 37.2985878, -14.9144278, 36.3317986, -51.5849495, 52.2130165

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4859338, upper bound: 57.4494298
time: 0.59 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4859338, upper bound: 57.5061026
time: 0.62 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -7.4350624, 32.4608459, -7.3273187, 31.7573109, -39.1923752, 39.7881660
1: -9.5189266, 36.8422890, -9.3588591, 36.0470428, -45.5659714, 46.2011490
2: -9.4125261, 36.2255974, -9.2744541, 35.4336548, -44.8461800, 45.5000534
3: -16.3478165, 39.4466934, -16.0628605, 38.5980263, -54.9458427, 55.5095520
4: -15.2531500, 37.2985878, -15.1160593, 36.5008430, -51.7539940, 52.4146423

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A1_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4936084, upper bound: 57.5073163
time: 0.61 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5046515, upper bound: 57.5120444
time: 0.53 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -6.2563448, 28.6933155, -7.7548656, 34.3289261, -40.5852661, 36.4481812
1: -7.9915113, 32.5977592, -9.9109211, 38.8927612, -46.8842735, 42.5086708
2: -7.9890079, 31.8166008, -9.8113804, 38.3144073, -46.3034096, 41.6279716
3: -14.0460072, 34.9063377, -17.1326771, 41.5241203, -55.5701294, 52.0390167
4: -13.2832928, 32.6852341, -16.0652657, 39.3405418, -52.6238327, 48.7504997

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_A1_B2_B1_A1_B1

### Relational analysis result of IS_A1_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3870960, upper bound: 57.3824989
time: 0.57 seconds

## Relational analysis of IS_A1_A1_B2_B1_A1_B2

### Relational analysis result of IS_A1_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3870960, upper bound: 57.4155201
time: 0.62 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -7.4350624, 32.4608459, -7.7548656, 34.3289261, -41.7639885, 40.2157097
1: -9.5189266, 36.8422890, -9.9109211, 38.8927612, -48.4116898, 46.7532082
2: -9.4125261, 36.2255974, -9.8113804, 38.3144073, -47.7269325, 46.0369759
3: -16.3478165, 39.4466934, -17.1326771, 41.5241203, -57.8719330, 56.5793686
4: -15.2531500, 37.2985878, -16.0652657, 39.3405418, -54.5936928, 53.3638535

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B2_B1_A2_A1

### Relational analysis result of IS_A1_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3948422, upper bound: 57.3903751
time: 0.61 seconds

## Relational analysis of IS_A1_A1_B2_B1_A2_A2

### Relational analysis result of IS_A1_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3459512, upper bound: 57.3691220
time: 0.68 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -7.4272614, 32.2916946, -8.3423462, 35.8539047, -43.2811661, 40.6340408
1: -9.5002413, 36.6233063, -10.6513557, 40.6247215, -50.1249619, 47.2746620
2: -9.3774567, 36.0825157, -10.4964466, 40.1818428, -49.5592995, 46.5789642
3: -16.3603516, 39.1245842, -18.1694336, 43.3589249, -59.7192764, 57.2940178
4: -15.2274914, 37.1916122, -16.8388348, 41.3315392, -56.5590286, 54.0304413

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_A1_B2_B2_B1_A1

### Relational analysis result of IS_A1_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3870960, upper bound: 57.4345497
time: 0.80 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_A2

### Relational analysis result of IS_A1_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3870960, upper bound: 57.4554520
time: 0.65 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -7.4272614, 32.2916946, -8.6094341, 36.7576447, -44.1849060, 40.9011307
1: -9.5002413, 36.6233063, -10.9853382, 41.6540833, -51.1543236, 47.6086426
2: -9.3774567, 36.0825157, -10.8296928, 41.1556931, -50.5331497, 46.9122086
3: -16.3603516, 39.1245842, -18.6601658, 44.4872971, -60.8476486, 57.7847443
4: -15.2274914, 37.1916122, -17.3866043, 42.3157730, -57.5432587, 54.5782089

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_A1_B2_B2_B2_A1

### Relational analysis result of IS_A1_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3870960, upper bound: 57.4637673
time: 0.62 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2_A2

### Relational analysis result of IS_A1_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3870960, upper bound: 57.4827401
time: 0.58 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -5.9096947, 27.3499146, -6.5372963, 29.5482712, -35.4579620, 33.8872108
1: -7.5164185, 31.1457329, -8.3372393, 33.5832443, -41.0996628, 39.4829597
2: -7.6001306, 30.2378502, -8.3474846, 32.7798882, -40.3800163, 38.5853348
3: -13.2140646, 33.4282455, -14.5443096, 36.0342140, -49.2482758, 47.9725571
4: -12.7368975, 30.9877415, -13.8917007, 33.6436195, -46.3805161, 44.8794403

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_A2_B1_B1_A1_B1

### Relational analysis result of IS_A1_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4439078, upper bound: 57.4333312
time: 0.53 seconds

## Relational analysis of IS_A1_A2_B1_B1_A1_B2

### Relational analysis result of IS_A1_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4439078, upper bound: 57.4812687
time: 0.58 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -7.3467827, 32.0613708, -6.5372963, 29.5482712, -36.8950424, 38.5986633
1: -9.3942413, 36.4329567, -8.3372393, 33.5832443, -42.9774857, 44.7701836
2: -9.3245058, 35.7049446, -8.3474846, 32.7798882, -42.1043930, 44.0524254
3: -16.1041088, 39.0988007, -14.5443096, 36.0342140, -52.1383209, 53.6431084
4: -15.1873417, 36.7418594, -13.8917007, 33.6436195, -48.8309631, 50.6335564

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_A2_B1_B1_A2_A1

### Relational analysis result of IS_A1_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4614279, upper bound: 57.4942538
time: 0.57 seconds

## Relational analysis of IS_A1_A2_B1_B1_A2_A2

### Relational analysis result of IS_A1_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4812687, upper bound: 57.4942538
time: 0.57 seconds

## BFS IS instance: IS_A1_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -6.5545845, 28.8362465, -6.3773355, 28.4110374, -34.9656219, 35.2135811
1: -8.3718977, 32.7864456, -8.1524849, 32.3773079, -40.7492065, 40.9389305
2: -8.3435240, 32.0661774, -8.1604490, 31.5259762, -39.8694992, 40.2266235
3: -14.4547787, 35.1783791, -14.0450106, 34.8340721, -49.2888489, 49.2233887
4: -13.6787777, 33.0736771, -13.4756565, 32.4121132, -46.0908890, 46.5493202

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_A2_B1_B2_B1_A1

### Relational analysis result of IS_A1_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4715838, upper bound: 57.5195957
time: 0.55 seconds

## Relational analysis of IS_A1_A2_B1_B2_B1_A2

### Relational analysis result of IS_A1_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4715838, upper bound: 57.5326214
time: 0.68 seconds

## BFS IS instance: IS_A1_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -7.0450864, 30.6773186, -7.2364883, 31.4795399, -38.5246277, 37.9138069
1: -8.9984789, 34.8401260, -9.2547970, 35.7788582, -44.7773361, 44.0949249
2: -8.9331036, 34.1861649, -9.1867266, 35.0583305, -43.9914322, 43.3728905
3: -15.4721489, 37.3334732, -15.8404684, 38.4033241, -53.8754730, 53.1739388
4: -14.5863972, 35.2302628, -14.9461699, 36.0971260, -50.6835251, 50.1764297

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_A2_B1_B2_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5326215, upper bound: 57.5326214
time: 0.67 seconds

## Relational analysis of IS_A1_A2_B1_B2_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5326215, upper bound: 57.5326214
time: 0.57 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -5.9096947, 27.3499146, -8.2839651, 36.3554649, -42.2651596, 35.6338806
1: -7.5164185, 31.1457329, -10.5785780, 41.1595268, -48.6759453, 41.7243080
2: -7.6001306, 30.2378502, -10.4618778, 40.6369743, -48.2371063, 40.6997299
3: -13.2140646, 33.4282455, -18.2245331, 43.9165955, -57.1306572, 51.6527748
4: -12.7368975, 30.9877415, -17.0619030, 41.7284966, -54.4653931, 48.0496445

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_A2_B2_B1_A1_B1

### Relational analysis result of IS_A1_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3966763, upper bound: 57.3851447
time: 0.61 seconds

## Relational analysis of IS_A1_A2_B2_B1_A1_B2

### Relational analysis result of IS_A1_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3966763, upper bound: 57.4812686
time: 0.67 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -7.3467827, 32.0613708, -8.2839651, 36.3554649, -43.7022438, 40.3453369
1: -9.3942413, 36.4329567, -10.5785780, 41.1595268, -50.5537682, 47.0115356
2: -9.3245058, 35.7049446, -10.4618778, 40.6369743, -49.9614792, 46.1668205
3: -16.1041088, 39.0988007, -18.2245331, 43.9165955, -60.0207062, 57.3233261
4: -15.1873417, 36.7418594, -17.0619030, 41.7284966, -56.9158401, 53.8037643

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_A2_B2_B1_A2_A1

### Relational analysis result of IS_A1_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4627581, upper bound: 57.5042658
time: 0.60 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2_A2

### Relational analysis result of IS_A1_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4843160, upper bound: 57.5042658
time: 0.64 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -6.8744965, 30.0656357, -8.6963406, 36.2858658, -43.1603508, 38.7619743
1: -8.7779045, 34.1606102, -11.0196209, 41.1690025, -49.9468994, 45.1802292
2: -8.7287350, 33.4744568, -10.8917398, 40.6589241, -49.3876572, 44.3661919
3: -15.1171074, 36.6302528, -18.5529976, 44.0802841, -59.1973801, 55.1832428
4: -14.2977638, 34.4894257, -17.6058350, 41.6847610, -55.9825249, 52.0952530

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A2_B2_B2_B1_B1

### Relational analysis result of IS_A1_A2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5335269, upper bound: 57.5367441
time: 0.65 seconds

## Relational analysis of IS_A1_A2_B2_B2_B1_B2

### Relational analysis result of IS_A1_A2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5330228, upper bound: 57.5342633
time: 0.63 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -7.2452183, 31.4364815, -9.0601625, 38.3075333, -45.5527496, 40.4966431
1: -9.2526312, 35.6897430, -11.5555935, 43.3868027, -52.6394348, 47.2453384
2: -9.1748924, 35.0614204, -11.3703423, 42.9524994, -52.1273918, 46.4317627
3: -15.8903408, 38.2248116, -19.5706577, 46.3116646, -62.2020035, 57.7954597
4: -14.9646664, 36.1183052, -18.2063713, 44.1730919, -59.1377525, 54.3246727

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_A2_B2_B2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4623107, upper bound: 57.4564568
time: 0.61 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4623107, upper bound: 57.5326251
time: 0.96 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -8.0627384, 35.7259407, -6.1734071, 28.1130219, -36.1757584, 41.8993492
1: -10.3429604, 40.4529800, -7.8701830, 31.9741211, -42.3170815, 48.3231544
2: -10.1579933, 39.9914627, -7.8979273, 31.1287766, -41.2867661, 47.8893890
3: -17.8792992, 43.0726242, -13.7752228, 34.3274574, -52.2067566, 56.8478355
4: -16.5315685, 41.0846443, -13.2005091, 31.9362679, -48.4678345, 54.2851524

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_A1_B1_A1_B1_B1

### Relational analysis result of IS_A2_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3824989, upper bound: 57.3870960
time: 0.69 seconds

## Relational analysis of IS_A2_A1_B1_A1_B1_B2

### Relational analysis result of IS_A2_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3824989, upper bound: 57.3966763
time: 0.63 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -8.0627384, 35.7259407, -7.4468746, 32.3137131, -40.3764496, 43.1728134
1: -10.3429604, 40.4529800, -9.5282478, 36.7067146, -47.0496712, 49.9812279
2: -10.1579933, 39.9914627, -9.4350853, 36.0107231, -46.1687164, 49.4265480
3: -17.8792992, 43.0726242, -16.3048954, 39.3740845, -57.2533836, 59.3775101
4: -16.5315685, 41.0846443, -15.3575916, 37.0480614, -53.5796280, 56.4422379

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_A1_B1_A1_B2_B1

### Relational analysis result of IS_A2_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.1735952, upper bound: 57.4285250
time: 0.63 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2_B2

### Relational analysis result of IS_A2_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3851447, upper bound: 57.4525981
time: 0.64 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.0433788, 38.5837364, -7.2896957, 31.6561909, -40.6995697, 45.8734245
1: -11.5689020, 43.6965904, -9.3200359, 35.9118881, -47.4807892, 53.0166245
2: -11.3229685, 43.3287659, -9.2074223, 35.3617973, -46.6847649, 52.5361862
3: -19.6716652, 46.5399094, -16.0440216, 38.3914108, -58.0630760, 62.5839310
4: -18.0751343, 44.5710564, -14.9644852, 36.4487572, -54.5238914, 59.5355301

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 25

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_A1_B1_A2_B1_B1

### Relational analysis result of IS_A2_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4345497, upper bound: 57.3978074
time: 0.60 seconds

## Relational analysis of IS_A2_A1_B1_A2_B1_B2

### Relational analysis result of IS_A2_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4345498, upper bound: 57.4581299
time: 0.71 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.0433788, 38.5837364, -7.3673935, 31.9134426, -40.9568214, 45.9511261
1: -11.5689020, 43.6965904, -9.4090042, 36.2246475, -47.7935486, 53.1055908
2: -11.3229685, 43.3287659, -9.3240471, 35.6100235, -46.9329910, 52.6528091
3: -19.6716652, 46.5399094, -16.1484604, 38.7872200, -58.4588852, 62.6883698
4: -18.0751343, 44.5710564, -15.1940689, 36.6823387, -54.7574730, 59.7651138

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A2_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A2_A1_B1_A2_B2_B1

### Relational analysis result of IS_A2_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4493432, upper bound: 57.4563193
time: 0.59 seconds

## Relational analysis of IS_A2_A1_B1_A2_B2_B2

### Relational analysis result of IS_A2_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4527611, upper bound: 57.4593339
time: 0.65 seconds

## BFS IS instance: IS_A2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -8.0627384, 35.7259407, -7.7548656, 34.3289261, -42.3916626, 43.4808044
1: -10.3429604, 40.4529800, -9.9109211, 38.8927612, -49.2357216, 50.3638954
2: -10.1579933, 39.9914627, -9.8113804, 38.3144073, -48.4723969, 49.8028374
3: -17.8792992, 43.0726242, -17.1326771, 41.5241203, -59.4034195, 60.2052994
4: -16.5315685, 41.0846443, -16.0652657, 39.3405418, -55.8721085, 57.1499062

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3596898, upper bound: 57.3570668
time: 0.64 seconds

## Relational analysis of IS_A2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3444953, upper bound: 57.3521646
time: 0.65 seconds

## BFS IS instance: IS_A2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -8.0627384, 35.7259407, -8.6534843, 36.8002853, -44.8630219, 44.3794250
1: -10.3429604, 40.4529800, -11.0422096, 41.6996994, -52.0426559, 51.4951897
2: -10.1579933, 39.9914627, -10.8732853, 41.2166405, -51.3746300, 50.8647423
3: -17.8792992, 43.0726242, -18.7433796, 44.5373421, -62.4166374, 61.8160019
4: -16.5315685, 41.0846443, -17.4636860, 42.3764076, -58.9079628, 58.5483322

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_A1_B2_A1_B2_B1

### Relational analysis result of IS_A2_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.1735952, upper bound: 57.4279040
time: 0.61 seconds

## Relational analysis of IS_A2_A1_B2_A1_B2_B2

### Relational analysis result of IS_A2_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3851447, upper bound: 57.4525981
time: 0.64 seconds

## BFS IS instance: IS_A2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -9.0433788, 38.5837364, -7.3897734, 32.9540634, -41.9974442, 45.9735069
1: -11.5689020, 43.6965904, -9.4480724, 37.3490868, -48.9179878, 53.1446609
2: -11.3229685, 43.3287659, -9.3739195, 36.7447433, -48.0677109, 52.7026863
3: -19.6716652, 46.5399094, -16.3672638, 39.9006386, -59.5723038, 62.9071732
4: -18.0751343, 44.5710564, -15.3699245, 37.7339973, -55.8091316, 59.9409790

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4202860, upper bound: 57.3646893
time: 0.90 seconds

## Relational analysis of IS_A2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3919552, upper bound: 57.3588146
time: 2.31 seconds

## BFS IS instance: IS_A2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -9.0433788, 38.5837364, -8.6534843, 36.8002853, -45.8436661, 47.2372169
1: -11.5689020, 43.6965904, -11.0422096, 41.6996994, -53.2686005, 54.7388000
2: -11.3229685, 43.3287659, -10.8732853, 41.2166405, -52.5396042, 54.2020416
3: -19.6716652, 46.5399094, -18.7433796, 44.5373421, -64.2090073, 65.2832870
4: -18.0751343, 44.5710564, -17.4636860, 42.3764076, -60.4515343, 62.0347443

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_A1_B2_A2_B2_B1

### Relational analysis result of IS_A2_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4329543, upper bound: 57.4504141
time: 0.65 seconds

## Relational analysis of IS_A2_A1_B2_A2_B2_B2

### Relational analysis result of IS_A2_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4329543, upper bound: 57.4617000
time: 0.66 seconds

## BFS IS instance: IS_A2_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -7.4656463, 33.3557091, -9.1575928, 38.8744278, -46.3400726, 42.5133018
1: -9.5422745, 37.8093300, -11.6959438, 44.0212021, -53.5634766, 49.5052719
2: -9.4745121, 37.1891479, -11.4683819, 43.6604576, -53.1349640, 48.6575317
3: -16.5416107, 40.3914528, -19.9659348, 46.8748016, -63.4164124, 60.3573723
4: -15.5500689, 38.1762428, -18.4506874, 44.9865723, -60.5366402, 56.6269302

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_A2_B1_B1_A1_B1

### Relational analysis result of IS_A2_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4155201, upper bound: 57.4059562
time: 0.53 seconds

## Relational analysis of IS_A2_A2_B1_B1_A1_B2

### Relational analysis result of IS_A2_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4155201, upper bound: 57.4664976
time: 0.60 seconds

## BFS IS instance: IS_A2_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -8.7081556, 37.1425133, -7.4272614, 32.2916946, -40.9998512, 44.5697746
1: -11.1119413, 42.0899849, -9.5002413, 36.6233063, -47.7352486, 51.5902252
2: -10.9512491, 41.5937920, -9.3774567, 36.0825157, -47.0337639, 50.9712448
3: -18.8729000, 44.9509163, -16.3603516, 39.1245842, -57.9974823, 61.3112679
4: -17.5746422, 42.7677650, -15.2274914, 37.1916122, -54.7662544, 57.9952545

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_A2_B1_B1_A2_A1

### Relational analysis result of IS_A2_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4822170, upper bound: 57.4721651
time: 0.64 seconds

## Relational analysis of IS_A2_A2_B1_B1_A2_A2

### Relational analysis result of IS_A2_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4865724, upper bound: 57.4769667
time: 0.65 seconds

## BFS IS instance: IS_A2_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -7.4656463, 33.3557091, -8.8118200, 37.3577957, -44.8234406, 42.1675301
1: -9.5422745, 37.8093300, -11.2442751, 42.3229713, -51.8652420, 49.0536041
2: -9.4745121, 37.1891479, -11.0679474, 41.8824844, -51.3569946, 48.2570915
3: -16.5416107, 40.3914528, -19.1606712, 45.1869125, -61.7285156, 59.5521240
4: -15.5500689, 38.1762428, -17.8871288, 43.1344337, -58.6845016, 56.0633698

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_A2_B1_B2_A1_B1

### Relational analysis result of IS_A2_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4918280, upper bound: 57.4845372
time: 0.60 seconds

## Relational analysis of IS_A2_A2_B1_B2_A1_B2

### Relational analysis result of IS_A2_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4918280, upper bound: 57.5238990
time: 0.65 seconds

## BFS IS instance: IS_A2_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -8.7081556, 37.1425133, -7.3673935, 31.9134426, -40.6215973, 44.5099030
1: -11.1119413, 42.0899849, -9.4090042, 36.2246475, -47.3365898, 51.4989853
2: -10.9512491, 41.5937920, -9.3240471, 35.6100235, -46.5612717, 50.9178314
3: -18.8729000, 44.9509163, -16.1484604, 38.7872200, -57.6601181, 61.0993767
4: -17.5746422, 42.7677650, -15.1940689, 36.6823387, -54.2569809, 57.9618340

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_A2_B1_B2_A2_A1

### Relational analysis result of IS_A2_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5367500, upper bound: 57.5337611
time: 0.94 seconds

## Relational analysis of IS_A2_A2_B1_B2_A2_A2

### Relational analysis result of IS_A2_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5342686, upper bound: 57.5331873
time: 0.62 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -7.4656463, 33.3557091, -8.2914886, 36.3861198, -43.8517647, 41.6471977
1: -9.5422745, 37.8093300, -10.5880194, 41.1941528, -50.7364273, 48.3973503
2: -9.4745121, 37.1891479, -10.4712791, 40.6714973, -50.1460075, 47.6604271
3: -16.5416107, 40.3914528, -18.2408638, 43.9536514, -60.4952621, 58.6323128
4: -15.5500689, 38.1762428, -17.0769653, 41.7640762, -57.3141441, 55.2532043

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_A2_B2_B1_A1_B1

### Relational analysis result of IS_A2_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4067223, upper bound: 57.3880659
time: 0.91 seconds

## Relational analysis of IS_A2_A2_B2_B1_A1_B2

### Relational analysis result of IS_A2_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4067223, upper bound: 57.5121573
time: 0.96 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -8.7081556, 37.1425133, -8.2914886, 36.3861198, -45.0942764, 45.4339981
1: -11.1119413, 42.0899849, -10.5880194, 41.1941528, -52.3060951, 52.6780052
2: -10.9512491, 41.5937920, -10.4712791, 40.6714973, -51.6227455, 52.0650711
3: -18.8729000, 44.9509163, -18.2408638, 43.9536514, -62.8265533, 63.1917801
4: -17.5746422, 42.7677650, -17.0769653, 41.7640762, -59.3387146, 59.8447304

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_A2_B2_B1_A2_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4969522, upper bound: 57.5020399
time: 0.71 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4900378, upper bound: 57.5014999
time: 0.61 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -8.6222887, 36.7295456, -8.8100138, 37.5939674, -46.2162552, 45.5395584
1: -10.9891005, 41.5826263, -11.2688313, 42.5857544, -53.5748520, 52.8514557
2: -10.8409109, 41.1832390, -11.0355730, 42.1981087, -53.0390205, 52.2188110
3: -18.7244911, 44.3607635, -19.1602993, 45.3824959, -64.1069794, 63.5210419
4: -17.4421425, 42.4057846, -17.6283436, 43.3953476, -60.8374710, 60.0341263

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_A2_B2_B2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4067223, upper bound: 57.4411211
time: 0.70 seconds

## Relational analysis of IS_A2_A2_B2_B2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4067223, upper bound: 57.4591648
time: 0.68 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -8.6222887, 36.7295456, -8.7081556, 37.1425133, -45.7648010, 45.4377022
1: -10.9891005, 41.5826263, -11.1119413, 42.0899849, -53.0790863, 52.6945686
2: -10.8409109, 41.1832390, -10.9512491, 41.5937920, -52.4347000, 52.1344872
3: -18.7244911, 44.3607635, -18.8729000, 44.9509163, -63.6754074, 63.2336578
4: -17.4421425, 42.4057846, -17.5746422, 42.7677650, -60.2099037, 59.9804192

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A2_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_A2_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A2_A2_B2_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4645763, upper bound: 57.5332771
time: 0.64 seconds

## Relational analysis of IS_A2_A2_B2_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4685367, upper bound: 57.5371131
time: 0.63 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 6.17 seconds
IS_A1_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.4292205, upper bound: 57.4292205
IS_A1_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.4292205, upper bound: 57.4439078
IS_A1_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.3731779, upper bound: 57.4773069
IS_A1_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.4333312, upper bound: 57.4934621
IS_A1_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.4859338, upper bound: 57.4494298
IS_A1_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.4859338, upper bound: 57.5061026
IS_A1_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.4936084, upper bound: 57.5073163
IS_A1_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.5046515, upper bound: 57.5120444
IS_A1_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.3870960, upper bound: 57.3824989
IS_A1_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.3870960, upper bound: 57.4155201
IS_A1_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.3948422, upper bound: 57.3903751
IS_A1_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.3459512, upper bound: 57.3691220
IS_A1_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.3870960, upper bound: 57.4345497
IS_A1_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.3870960, upper bound: 57.4554520
IS_A1_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.3870960, upper bound: 57.4637673
IS_A1_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.3870960, upper bound: 57.4827401
IS_A1_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.4439078, upper bound: 57.4333312
IS_A1_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.4439078, upper bound: 57.4812687
IS_A1_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.4614279, upper bound: 57.4942538
IS_A1_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.4812687, upper bound: 57.4942538
IS_A1_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.4715838, upper bound: 57.5195957
IS_A1_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.4715838, upper bound: 57.5326214
IS_A1_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.5326215, upper bound: 57.5326214
IS_A1_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.5326215, upper bound: 57.5326214
IS_A1_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.3966763, upper bound: 57.3851447
IS_A1_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.3966763, upper bound: 57.4812686
IS_A1_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.4627581, upper bound: 57.5042658
IS_A1_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.4843160, upper bound: 57.5042658
IS_A1_A2_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.5335269, upper bound: 57.5367441
IS_A1_A2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.5330228, upper bound: 57.5342633
IS_A1_A2_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.4623107, upper bound: 57.4564568
IS_A1_A2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.4623107, upper bound: 57.5326251
IS_A2_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.3824989, upper bound: 57.3870960
IS_A2_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.3824989, upper bound: 57.3966763
IS_A2_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.1735952, upper bound: 57.4285250
IS_A2_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.3851447, upper bound: 57.4525981
IS_A2_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.4345497, upper bound: 57.3978074
IS_A2_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.4345498, upper bound: 57.4581299
IS_A2_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.4493432, upper bound: 57.4563193
IS_A2_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.4527611, upper bound: 57.4593339
IS_A2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.3596898, upper bound: 57.3570668
IS_A2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.3444953, upper bound: 57.3521646
IS_A2_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.1735952, upper bound: 57.4279040
IS_A2_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.3851447, upper bound: 57.4525981
IS_A2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.4202860, upper bound: 57.3646893
IS_A2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.3919552, upper bound: 57.3588146
IS_A2_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.4329543, upper bound: 57.4504141
IS_A2_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.4329543, upper bound: 57.4617000
IS_A2_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.4155201, upper bound: 57.4059562
IS_A2_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.4155201, upper bound: 57.4664976
IS_A2_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.4822170, upper bound: 57.4721651
IS_A2_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.4865724, upper bound: 57.4769667
IS_A2_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.4918280, upper bound: 57.4845372
IS_A2_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.4918280, upper bound: 57.5238990
IS_A2_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.5367500, upper bound: 57.5337611
IS_A2_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.5342686, upper bound: 57.5331873
IS_A2_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.4067223, upper bound: 57.3880659
IS_A2_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.4067223, upper bound: 57.5121573
IS_A2_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.4969522, upper bound: 57.5020399
IS_A2_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.4900378, upper bound: 57.5014999
IS_A2_A2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.4067223, upper bound: 57.4411211
IS_A2_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.4067223, upper bound: 57.4591648
IS_A2_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.4645763, upper bound: 57.5332771
IS_A2_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.17
Output dim: 0, lower bound: -57.4685367, upper bound: 57.5371131

## BFS IS instance: IS_A1_A1_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -6.2563448, 28.6933155, -6.2028508, 28.4855404, -34.7418785, 34.8961601
1: -7.9915113, 32.5977592, -7.9241486, 32.3644218, -40.3559303, 40.5219040
2: -7.9890079, 31.8166008, -7.9234486, 31.5827866, -39.5717850, 39.7400436
3: -14.0460072, 34.9063377, -13.9296494, 34.6559029, -48.7019119, 48.8359871
4: -13.2832928, 32.6852341, -13.1778812, 32.4421234, -45.7254143, 45.8631134

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_A1_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_A1_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A1_B1_B1_A1

### Relational analysis result of IS_A1_A1_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3756167, upper bound: 57.3472274
time: 0.76 seconds

## Relational analysis of IS_A1_A1_B1_A1_B1_B1_A2

### Relational analysis result of IS_A1_A1_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3332711, upper bound: 57.3332711
time: 0.57 seconds

## BFS IS instance: IS_A1_A1_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -6.2563448, 28.6933155, -5.8524818, 27.1140995, -33.3704453, 34.5457993
1: -7.9915113, 32.5977592, -7.4440780, 30.8827305, -38.8742409, 40.0418243
2: -7.9890079, 31.8166008, -7.5294771, 29.9740219, -37.9630241, 39.3460655
3: -14.0460072, 34.9063377, -13.0862598, 33.1450462, -47.1910553, 47.9925957
4: -13.2832928, 32.6852341, -12.6227808, 30.7116623, -43.9949493, 45.3080139

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A1_B1_B2_A1

### Relational analysis result of IS_A1_A1_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3756167, upper bound: 57.3849383
time: 0.58 seconds

## Relational analysis of IS_A1_A1_B1_A1_B1_B2_A2

### Relational analysis result of IS_A1_A1_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3332711, upper bound: 57.3625311
time: 0.60 seconds

## BFS IS instance: IS_A1_A1_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -5.4232903, 25.5673122, -6.1016788, 27.3407936, -32.7640839, 31.6689911
1: -6.9047184, 29.1354637, -7.7967114, 31.1733246, -38.0780411, 36.9321747
2: -6.9951367, 28.2539806, -7.8216729, 30.2965794, -37.2917099, 36.0756531
3: -12.2459021, 31.2145977, -13.4525700, 33.5597610, -45.8056641, 44.6671638
4: -11.7124987, 29.0004520, -12.9597168, 31.1350784, -42.8475761, 41.9601593

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_A1_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_A1_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_A1_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_A1_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1_A1_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of IS_A1_A1_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A1_A1_B1_A1_B2_B1_B1

### Relational analysis result of IS_A1_A1_B1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3984789, upper bound: 57.4707923
time: 0.62 seconds

## Relational analysis of IS_A1_A1_B1_A1_B2_B1_B2

### Relational analysis result of IS_A1_A1_B1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4030430, upper bound: 57.4735317
time: 0.85 seconds

## BFS IS instance: IS_A1_A1_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -5.9418092, 27.4840260, -6.7143097, 29.5272770, -35.4690857, 34.1983337
1: -7.5859861, 31.2516060, -8.5912838, 33.5915833, -41.1775703, 39.8428879
2: -7.6131682, 30.4371834, -8.5521288, 32.8054466, -40.4186134, 38.9893036
3: -13.3715000, 33.4818764, -14.7602081, 36.0950890, -49.4665909, 48.2420807
4: -12.6838436, 31.2686234, -13.9919653, 33.7611427, -46.4449844, 45.2605820

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_A1_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_A1_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_A1_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_A1_B1_A1_B2_B2_B1

### Relational analysis result of IS_A1_A1_B1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4494298, upper bound: 57.4859338
time: 0.64 seconds

## Relational analysis of IS_A1_A1_B1_A1_B2_B2_B2

### Relational analysis result of IS_A1_A1_B1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4494298, upper bound: 57.4934621
time: 0.58 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -7.4350624, 32.4608459, -5.7745767, 26.6899109, -34.1249733, 38.2354126
1: -9.5189266, 36.8422890, -7.3628654, 30.3618813, -39.8808060, 44.2051544
2: -9.4125261, 36.2255974, -7.4123101, 29.5563755, -38.9689026, 43.6379089
3: -16.3478165, 39.4466934, -12.9636087, 32.5635490, -48.9113655, 52.4102936
4: -15.2531500, 37.2985878, -12.3595028, 30.3521996, -45.6053505, 49.6580887

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_A1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_A1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B1_B1_A1

### Relational analysis result of IS_A1_A1_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4502422, upper bound: 57.3755092
time: 0.60 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_B1_A2

### Relational analysis result of IS_A1_A1_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3991324, upper bound: 57.3474057
time: 0.67 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.63 seconds
IS_A1_A1_B1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 0, lower bound: -57.3756167, upper bound: 57.3472274
IS_A1_A1_B1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 0, lower bound: -57.3332711, upper bound: 57.3332711
IS_A1_A1_B1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 0, lower bound: -57.3756167, upper bound: 57.3849383
IS_A1_A1_B1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 0, lower bound: -57.3332711, upper bound: 57.3625311
IS_A1_A1_B1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 0, lower bound: -57.3984789, upper bound: 57.4707923
IS_A1_A1_B1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 0, lower bound: -57.4030430, upper bound: 57.4735317
IS_A1_A1_B1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 0, lower bound: -57.4494298, upper bound: 57.4859338
IS_A1_A1_B1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 0, lower bound: -57.4494298, upper bound: 57.4934621
IS_A1_A1_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 0, lower bound: -57.4502422, upper bound: 57.3755092
IS_A1_A1_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.63
Output dim: 0, lower bound: -57.3991324, upper bound: 57.3474057
IS_A1_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.4859338, upper bound: 57.5061026
IS_A1_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.4936084, upper bound: 57.5073163
IS_A1_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.5046515, upper bound: 57.5120444
IS_A1_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.3870960, upper bound: 57.3824989
IS_A1_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.3870960, upper bound: 57.4155201
IS_A1_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.3948422, upper bound: 57.3903751
IS_A1_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.3459512, upper bound: 57.3691220
IS_A1_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.3870960, upper bound: 57.4345497
IS_A1_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.3870960, upper bound: 57.4554520
IS_A1_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.3870960, upper bound: 57.4637673
IS_A1_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.3870960, upper bound: 57.4827401
IS_A1_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.4439078, upper bound: 57.4333312
IS_A1_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.4439078, upper bound: 57.4812687
IS_A1_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.4614279, upper bound: 57.4942538
IS_A1_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.4812687, upper bound: 57.4942538
IS_A1_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.4715838, upper bound: 57.5195957
IS_A1_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.4715838, upper bound: 57.5326214
IS_A1_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.5326215, upper bound: 57.5326214
IS_A1_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.5326215, upper bound: 57.5326214
IS_A1_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.3966763, upper bound: 57.3851447
IS_A1_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.3966763, upper bound: 57.4812686
IS_A1_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.4627581, upper bound: 57.5042658
IS_A1_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.4843160, upper bound: 57.5042658
IS_A1_A2_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.5335269, upper bound: 57.5367441
IS_A1_A2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.5330228, upper bound: 57.5342633
IS_A1_A2_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.4623107, upper bound: 57.4564568
IS_A1_A2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.4623107, upper bound: 57.5326251
IS_A2_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.3824989, upper bound: 57.3870960
IS_A2_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.3824989, upper bound: 57.3966763
IS_A2_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.1735952, upper bound: 57.4285250
IS_A2_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.3851447, upper bound: 57.4525981
IS_A2_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.4345497, upper bound: 57.3978074
IS_A2_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.4345498, upper bound: 57.4581299
IS_A2_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.4493432, upper bound: 57.4563193
IS_A2_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.4527611, upper bound: 57.4593339
IS_A2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.3596898, upper bound: 57.3570668
IS_A2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.3444953, upper bound: 57.3521646
IS_A2_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.1735952, upper bound: 57.4279040
IS_A2_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.3851447, upper bound: 57.4525981
IS_A2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.4202860, upper bound: 57.3646893
IS_A2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.3919552, upper bound: 57.3588146
IS_A2_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.4329543, upper bound: 57.4504141
IS_A2_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.4329543, upper bound: 57.4617000
IS_A2_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.4155201, upper bound: 57.4059562
IS_A2_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.4155201, upper bound: 57.4664976
IS_A2_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.4822170, upper bound: 57.4721651
IS_A2_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.4865724, upper bound: 57.4769667
IS_A2_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.4918280, upper bound: 57.4845372
IS_A2_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.4918280, upper bound: 57.5238990
IS_A2_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.5367500, upper bound: 57.5337611
IS_A2_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.5342686, upper bound: 57.5331873
IS_A2_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.4067223, upper bound: 57.3880659
IS_A2_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.4067223, upper bound: 57.5121573
IS_A2_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.4969522, upper bound: 57.5020399
IS_A2_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.4900378, upper bound: 57.5014999
IS_A2_A2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.4067223, upper bound: 57.4411211
IS_A2_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.4067223, upper bound: 57.4591648
IS_A2_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.4645763, upper bound: 57.5332771
IS_A2_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.63
Output dim: 0, lower bound: -57.4685367, upper bound: 57.5371131
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=66.57380676269531
rel_dist={0: [-57.5686962838552, 57.5686962838552]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 1125.96 seconds
