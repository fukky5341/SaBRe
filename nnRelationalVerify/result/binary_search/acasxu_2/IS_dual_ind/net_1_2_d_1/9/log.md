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
execution time: IAR + LP analysis = 2.10 + 1.62 = 3.71 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -57.5687468, upper bound: 57.5687468


# Binary Search by BASE starts (time budget: 1196.29 seconds, max iter: 100)

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
Binary search time: 71.68 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 1124.60 seconds

## Binary search (step 0) starts
Candidate diff: 0.2500000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5543357, upper bound: 57.5591333
time: 0.54 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5621985, upper bound: 57.5621985
time: 0.54 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.27 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.27
Output dim: 0, lower bound: -57.5543357, upper bound: 57.5591333
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.27
Output dim: 0, lower bound: -57.5621985, upper bound: 57.5621985

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -10.1770201, 42.4380188, -12.8046780, 52.5998383, -62.7768593, 55.2426949
1: -12.9492474, 48.0443954, -16.2452068, 59.4991722, -72.4484177, 64.2896042
2: -12.7204800, 47.7395439, -15.9143915, 59.4335175, -72.1539993, 63.6539268
3: -21.9290562, 51.2056084, -27.3275928, 63.1831360, -85.1121902, 78.5332031
4: -20.4464874, 49.1660233, -25.3546982, 61.2416534, -81.6881332, 74.5207214

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5512705, upper bound: 57.5512705
time: 0.49 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5512705, upper bound: 57.5591333
time: 0.64 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -11.9793425, 49.4708633, -13.0451412, 53.5286636, -65.5080032, 62.5160027
1: -15.2106190, 55.9559326, -16.5472050, 60.5470352, -75.7576523, 72.5031357
2: -14.9106045, 55.8369446, -16.2069473, 60.5032959, -75.4138870, 72.0438919
3: -25.6317139, 59.4555740, -27.8260193, 64.2862701, -89.9179688, 87.2815933
4: -23.7928352, 57.5261345, -25.8074226, 62.3490105, -86.1418457, 83.3335419

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5591333, upper bound: 57.5543357
time: 0.94 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5591333, upper bound: 57.5621985
time: 1.10 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.22 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.22
Output dim: 0, lower bound: -57.5512705, upper bound: 57.5512705
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.22
Output dim: 0, lower bound: -57.5512705, upper bound: 57.5591333
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.22
Output dim: 0, lower bound: -57.5591333, upper bound: 57.5543357
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.22
Output dim: 0, lower bound: -57.5591333, upper bound: 57.5621985

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -10.1770201, 42.4380188, -10.1770201, 42.4380188, -52.6150398, 52.6150398
1: -12.9492474, 48.0443954, -12.9492474, 48.0443954, -60.9936447, 60.9936447
2: -12.7204800, 47.7395439, -12.7204800, 47.7395439, -60.4600182, 60.4600143
3: -21.9290562, 51.2056084, -21.9290562, 51.2056084, -73.1346588, 73.1346588
4: -20.4464874, 49.1660233, -20.4464874, 49.1660233, -69.6124954, 69.6124954

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5258817, upper bound: 57.5440987
time: 0.52 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5512705, upper bound: 57.5512705
time: 0.56 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -10.1770201, 42.4380188, -11.9793425, 49.4708633, -59.6478844, 54.4173622
1: -12.9492474, 48.0443954, -15.2106190, 55.9559326, -68.9051743, 63.2550125
2: -12.7204800, 47.7395439, -14.9106045, 55.8369446, -68.5574265, 62.6501427
3: -21.9290562, 51.2056084, -25.6317139, 59.4555740, -81.3846283, 76.8373108
4: -20.4464874, 49.1660233, -23.7928352, 57.5261345, -77.9726028, 72.9588547

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5258817, upper bound: 57.5445685
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5512705, upper bound: 57.5591333
time: 0.53 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -11.9793425, 49.4708633, -10.1770201, 42.4380188, -54.4173622, 59.6478844
1: -15.2106190, 55.9559326, -12.9492474, 48.0443954, -63.2550125, 68.9051743
2: -14.9106045, 55.8369446, -12.7204800, 47.7395439, -62.6501465, 68.5574265
3: -25.6317139, 59.4555740, -21.9290562, 51.2056084, -76.8373108, 81.3846283
4: -23.7928352, 57.5261345, -20.4464874, 49.1660233, -72.9588547, 77.9725952

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4881302, upper bound: 57.5134218
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5591333, upper bound: 57.5543357
time: 0.66 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -11.9793425, 49.4708633, -11.9793425, 49.4708633, -61.4502068, 61.4502068
1: -15.2106190, 55.9559326, -15.2106190, 55.9559326, -71.1665497, 71.1665497
2: -14.9106045, 55.8369446, -14.9106045, 55.8369446, -70.7475510, 70.7475510
3: -25.6317139, 59.4555740, -25.6317139, 59.4555740, -85.0872879, 85.0872879
4: -23.7928352, 57.5261345, -23.7928352, 57.5261345, -81.3189545, 81.3189621

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4881302, upper bound: 57.5134218
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5591333, upper bound: 57.5621587
time: 0.57 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.38 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.38
Output dim: 0, lower bound: -57.5258817, upper bound: 57.5440987
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.38
Output dim: 0, lower bound: -57.5512705, upper bound: 57.5512705
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.38
Output dim: 0, lower bound: -57.5258817, upper bound: 57.5445685
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.38
Output dim: 0, lower bound: -57.5512705, upper bound: 57.5591333
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.38
Output dim: 0, lower bound: -57.4881302, upper bound: 57.5134218
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.38
Output dim: 0, lower bound: -57.5591333, upper bound: 57.5543357
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.38
Output dim: 0, lower bound: -57.4881302, upper bound: 57.5134218
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.38
Output dim: 0, lower bound: -57.5591333, upper bound: 57.5621587

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -9.1575928, 38.8744278, -10.0502205, 41.9608383, -51.1184273, 48.9246445
1: -11.6959438, 44.0212021, -12.7906437, 47.5061188, -59.2020645, 56.8118439
2: -11.4683819, 43.6604576, -12.5653629, 47.1894112, -58.6577911, 56.2258148
3: -19.9659348, 46.8748016, -21.6718063, 50.6372032, -70.6031342, 68.5466080
4: -18.4506874, 44.9865723, -20.2068501, 48.5900345, -67.0407257, 65.1934204

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5189899, upper bound: 57.5189899
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5189899, upper bound: 57.5440987
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8.8118200, 37.3577957, -10.1770201, 42.4380188, -51.2498398, 47.5348167
1: -11.2442751, 42.3229713, -12.9492474, 48.0443954, -59.2886696, 55.2722168
2: -11.0679474, 41.8824844, -12.7204800, 47.7395439, -58.8074799, 54.6029587
3: -19.1606712, 45.1869125, -21.9290562, 51.2056084, -70.3662796, 67.1159668
4: -17.8871288, 43.1344337, -20.4464874, 49.1660233, -67.0531387, 63.5809097

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5440987, upper bound: 57.5258817
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5440987, upper bound: 57.5512705
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -9.1575928, 38.8744278, -11.8326950, 48.9171371, -58.0747185, 50.7071190
1: -11.6959438, 44.0212021, -15.0274706, 55.3317108, -67.0276566, 59.0486717
2: -11.4683819, 43.6604576, -14.7309437, 55.2015610, -66.6699371, 58.3913994
3: -19.9659348, 46.8748016, -25.3336296, 58.7973213, -78.7632599, 72.2084351
4: -18.4506874, 44.9865723, -23.5139236, 56.8649902, -75.3156738, 68.5004959

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4947674, upper bound: 57.4830585
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4947674, upper bound: 57.5445685
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.8118200, 37.3577957, -11.9793425, 49.4708633, -58.2826843, 49.3371353
1: -11.2442751, 42.3229713, -15.2106190, 55.9559326, -67.2002029, 57.5335922
2: -11.0679474, 41.8824844, -14.9106045, 55.8369446, -66.9048920, 56.7930908
3: -19.1606712, 45.1869125, -25.6317139, 59.4555740, -78.6162415, 70.8186111
4: -17.8871288, 43.1344337, -23.7928352, 57.5261345, -75.4132385, 66.9272690

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5134218, upper bound: 57.4881302
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5134218, upper bound: 57.5591333
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -11.0297852, 46.1416092, -10.0502205, 41.9608383, -52.9906235, 56.1918259
1: -14.0756989, 52.2037125, -12.7906437, 47.5061188, -61.5818176, 64.9943542
2: -13.7119274, 52.0922241, -12.5653629, 47.1894112, -60.9013367, 64.6575851
3: -23.8188839, 55.4197121, -21.6718063, 50.6372032, -74.4560852, 77.0915222
4: -21.8664837, 53.5777702, -20.2068501, 48.5900345, -70.4565201, 73.7846222

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4830585, upper bound: 57.4947674
time: 0.51 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4830585, upper bound: 57.5134218
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -10.4046993, 43.4960098, -10.1770201, 42.4380188, -52.8427162, 53.6730309
1: -13.2431650, 49.2144547, -12.9492474, 48.0443954, -61.2875595, 62.1637039
2: -12.9902172, 48.9800606, -12.7204800, 47.7395439, -60.7297478, 61.7005386
3: -22.4255047, 52.3740425, -21.9290562, 51.2056084, -73.6311111, 74.3030853
4: -20.8130035, 50.4081459, -20.4464874, 49.1660233, -69.9790115, 70.8546066

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5445685, upper bound: 57.5260418
time: 0.50 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5445685, upper bound: 57.5543357
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -11.0297852, 46.1416092, -11.8326950, 48.9171371, -59.9469223, 57.9743004
1: -14.0756989, 52.2037125, -15.0274706, 55.3317108, -69.4074097, 67.2311859
2: -13.7119274, 52.0922241, -14.7309437, 55.2015610, -68.9134674, 66.8231659
3: -23.8188839, 55.4197121, -25.3336296, 58.7973213, -82.6162033, 80.7533417
4: -21.8664837, 53.5777702, -23.5139236, 56.8649902, -78.7314758, 77.0916901

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4669328, upper bound: 57.4669328
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4669328, upper bound: 57.5134218
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -10.4046993, 43.4960098, -11.9793425, 49.4708633, -59.8755646, 55.4753494
1: -13.2431650, 49.2144547, -15.2106190, 55.9559326, -69.1990967, 64.4250717
2: -12.9902172, 48.9800606, -14.9106045, 55.8369446, -68.8271637, 63.8906631
3: -22.4255047, 52.3740425, -25.6317139, 59.4555740, -81.8810806, 78.0057449
4: -20.8130035, 50.4081459, -23.7928352, 57.5261345, -78.3391266, 74.2009735

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5243553, upper bound: 57.4917810
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5243553, upper bound: 57.5621587
time: 0.54 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.33 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -57.5189899, upper bound: 57.5189899
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -57.5189899, upper bound: 57.5440987
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -57.5440987, upper bound: 57.5258817
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -57.5440987, upper bound: 57.5512705
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -57.4947674, upper bound: 57.4830585
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -57.4947674, upper bound: 57.5445685
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -57.5134218, upper bound: 57.4881302
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -57.5134218, upper bound: 57.5591333
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -57.4830585, upper bound: 57.4947674
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -57.4830585, upper bound: 57.5134218
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -57.5445685, upper bound: 57.5260418
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -57.5445685, upper bound: 57.5543357
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -57.4669328, upper bound: 57.4669328
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -57.4669328, upper bound: 57.5134218
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -57.5243553, upper bound: 57.4917810
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -57.5243553, upper bound: 57.5621587

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -9.1575928, 38.8744278, -9.1575928, 38.8744278, -48.0320168, 48.0320168
1: -11.6959438, 44.0212021, -11.6959438, 44.0212021, -55.7171478, 55.7171478
2: -11.4683819, 43.6604576, -11.4683819, 43.6604576, -55.1288376, 55.1288376
3: -19.9659348, 46.8748016, -19.9659348, 46.8748016, -66.8407288, 66.8407288
4: -18.4506874, 44.9865723, -18.4506874, 44.9865723, -63.4372597, 63.4372597

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5138670, upper bound: 57.5125661
time: 0.50 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5087887, upper bound: 57.5087887
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -9.1575928, 38.8744278, -8.8118200, 37.3577957, -46.5153885, 47.6862488
1: -11.6959438, 44.0212021, -11.2442751, 42.3229713, -54.0189133, 55.2654762
2: -11.4683819, 43.6604576, -11.0679474, 41.8824844, -53.3508682, 54.7284012
3: -19.9659348, 46.8748016, -19.1606712, 45.1869125, -65.1528397, 66.0354691
4: -18.4506874, 44.9865723, -17.8871288, 43.1344337, -61.5851212, 62.8737030

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5138670, upper bound: 57.5427675
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5087887, upper bound: 57.5392032
time: 0.52 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.8118200, 37.3577957, -9.1575928, 38.8744278, -47.6862488, 46.5153885
1: -11.2442751, 42.3229713, -11.6959438, 44.0212021, -55.2654762, 54.0189133
2: -11.0679474, 41.8824844, -11.4683819, 43.6604576, -54.7284012, 53.3508682
3: -19.1606712, 45.1869125, -19.9659348, 46.8748016, -66.0354691, 65.1528397
4: -17.8871288, 43.1344337, -18.4506874, 44.9865723, -62.8737030, 61.5851212

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5226232, upper bound: 57.5141803
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5392032, upper bound: 57.5192874
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.8118200, 37.3577957, -8.8118200, 37.3577957, -46.1696167, 46.1696167
1: -11.2442751, 42.3229713, -11.2442751, 42.3229713, -53.5672455, 53.5672455
2: -11.0679474, 41.8824844, -11.0679474, 41.8824844, -52.9504280, 52.9504280
3: -19.1606712, 45.1869125, -19.1606712, 45.1869125, -64.3475800, 64.3475800
4: -17.8871288, 43.1344337, -17.8871288, 43.1344337, -61.0215607, 61.0215607

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5226232, upper bound: 57.5280147
time: 0.51 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5392032, upper bound: 57.5497019
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.1575928, 38.8744278, -11.0297852, 46.1416092, -55.2991982, 49.9042091
1: -11.6959438, 44.0212021, -14.0756989, 52.2037125, -63.8996582, 58.0969009
2: -11.4683819, 43.6604576, -13.7119274, 52.0922241, -63.5606079, 57.3723793
3: -19.9659348, 46.8748016, -23.8188839, 55.4197121, -75.3856506, 70.6936874
4: -18.4506874, 44.9865723, -21.8664837, 53.5777702, -72.0284576, 66.8530579

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4927600, upper bound: 57.4817023
time: 0.51 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4932159, upper bound: 57.4818414
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.1575928, 38.8744278, -10.4046993, 43.4960098, -52.6535950, 49.2791214
1: -11.6959438, 44.0212021, -13.2431650, 49.2144547, -60.9104004, 57.2643661
2: -11.4683819, 43.6604576, -12.9902172, 48.9800606, -60.4484406, 56.6506729
3: -19.9659348, 46.8748016, -22.4255047, 52.3740425, -72.3399658, 69.3003082
4: -18.4506874, 44.9865723, -20.8130035, 50.4081459, -68.8588257, 65.7995758

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4927600, upper bound: 57.5432691
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4932159, upper bound: 57.5422614
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.8118200, 37.3577957, -11.0297852, 46.1416092, -54.9534302, 48.3875809
1: -11.2442751, 42.3229713, -14.0756989, 52.2037125, -63.4479866, 56.3986702
2: -11.0679474, 41.8824844, -13.7119274, 52.0922241, -63.1601715, 55.5944099
3: -19.1606712, 45.1869125, -23.8188839, 55.4197121, -74.5803833, 69.0057983
4: -17.8871288, 43.1344337, -21.8664837, 53.5777702, -71.4648895, 65.0009155

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4926662, upper bound: 57.4803831
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5130679, upper bound: 57.4875085
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.8118200, 37.3577957, -10.4046993, 43.4960098, -52.3078308, 47.7624931
1: -11.2442751, 42.3229713, -13.2431650, 49.2144547, -60.4587288, 55.5661354
2: -11.0679474, 41.8824844, -12.9902172, 48.9800606, -60.0480080, 54.8726997
3: -19.1606712, 45.1869125, -22.4255047, 52.3740425, -71.5347137, 67.6124191
4: -17.8871288, 43.1344337, -20.8130035, 50.4081459, -68.2952499, 63.9474373

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4926662, upper bound: 57.5079981
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5130679, upper bound: 57.5497019
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -11.0297852, 46.1416092, -9.1575928, 38.8744278, -49.9042130, 55.2991982
1: -14.0756989, 52.2037125, -11.6959438, 44.0212021, -58.0969009, 63.8996582
2: -13.7119274, 52.0922241, -11.4683819, 43.6604576, -57.3723831, 63.5606079
3: -23.8188839, 55.4197121, -19.9659348, 46.8748016, -70.6936874, 75.3856506
4: -21.8664837, 53.5777702, -18.4506874, 44.9865723, -66.8530579, 72.0284576

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.1475984, upper bound: 57.3365275
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4818414, upper bound: 57.4932159
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -11.0297852, 46.1416092, -8.8118200, 37.3577957, -48.3875809, 54.9534302
1: -14.0756989, 52.2037125, -11.2442751, 42.3229713, -56.3986702, 63.4479866
2: -13.7119274, 52.0922241, -11.0679474, 41.8824844, -55.5944099, 63.1601715
3: -23.8188839, 55.4197121, -19.1606712, 45.1869125, -69.0057983, 74.5803833
4: -21.8664837, 53.5777702, -17.8871288, 43.1344337, -65.0009155, 71.4648895

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.1475984, upper bound: 57.3669419
time: 0.50 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4818414, upper bound: 57.5130679
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -10.4046993, 43.4960098, -9.1575928, 38.8744278, -49.2791176, 52.6535950
1: -13.2431650, 49.2144547, -11.6959438, 44.0212021, -57.2643661, 60.9104004
2: -12.9902172, 48.9800606, -11.4683819, 43.6604576, -56.6506729, 60.4484406
3: -22.4255047, 52.3740425, -19.9659348, 46.8748016, -69.3003006, 72.3399658
4: -20.8130035, 50.4081459, -18.4506874, 44.9865723, -65.7995758, 68.8588257

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2967605, upper bound: 57.4165580
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5422614, upper bound: 57.5202924
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -10.4046993, 43.4960098, -8.8118200, 37.3577957, -47.7624893, 52.3078308
1: -13.2431650, 49.2144547, -11.2442751, 42.3229713, -55.5661354, 60.4587288
2: -12.9902172, 48.9800606, -11.0679474, 41.8824844, -54.8726997, 60.0480080
3: -22.4255047, 52.3740425, -19.1606712, 45.1869125, -67.6124115, 71.5347061
4: -20.8130035, 50.4081459, -17.8871288, 43.1344337, -63.9474335, 68.2952499

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2967605, upper bound: 57.4338251
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5422614, upper bound: 57.5533177
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -11.0297852, 46.1416092, -11.0297852, 46.1416092, -57.1713943, 57.1713943
1: -14.0756989, 52.2037125, -14.0756989, 52.2037125, -66.2794113, 66.2794113
2: -13.7119274, 52.0922241, -13.7119274, 52.0922241, -65.8041458, 65.8041458
3: -23.8188839, 55.4197121, -23.8188839, 55.4197121, -79.2385941, 79.2385941
4: -21.8664837, 53.5777702, -21.8664837, 53.5777702, -75.4442520, 75.4442520

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.1322288, upper bound: 57.3096871
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4665788, upper bound: 57.4665788
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -11.0297852, 46.1416092, -10.4046993, 43.4960098, -54.5257912, 56.5463028
1: -14.0756989, 52.2037125, -13.2431650, 49.2144547, -63.2901535, 65.4468765
2: -13.7119274, 52.0922241, -12.9902172, 48.9800606, -62.6919861, 65.0824432
3: -23.8188839, 55.4197121, -22.4255047, 52.3740425, -76.1929245, 77.8452148
4: -21.8664837, 53.5777702, -20.8130035, 50.4081459, -72.2746277, 74.3907700

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.1322288, upper bound: 57.3669419
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4665788, upper bound: 57.5130679
time: 0.52 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -10.4046993, 43.4960098, -11.0297852, 46.1416092, -56.5463028, 54.5257950
1: -13.2431650, 49.2144547, -14.0756989, 52.2037125, -65.4468765, 63.2901535
2: -12.9902172, 48.9800606, -13.7119274, 52.0922241, -65.0824432, 62.6919861
3: -22.4255047, 52.3740425, -23.8188839, 55.4197121, -77.8452148, 76.1929245
4: -20.8130035, 50.4081459, -21.8664837, 53.5777702, -74.3907700, 72.2746277

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2555014, upper bound: 57.3391795
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5240013, upper bound: 57.4916812
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -10.4046993, 43.4960098, -10.4046993, 43.4960098, -53.9007034, 53.9006996
1: -13.2431650, 49.2144547, -13.2431650, 49.2144547, -62.4576187, 62.4576187
2: -12.9902172, 48.9800606, -12.9902172, 48.9800606, -61.9702759, 61.9702644
3: -22.4255047, 52.3740425, -22.4255047, 52.3740425, -74.7995453, 74.7995453
4: -20.8130035, 50.4081459, -20.8130035, 50.4081459, -71.2211304, 71.2211304

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2555014, upper bound: 57.4176680
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5240013, upper bound: 57.5620557
time: 0.61 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.48 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -57.5138670, upper bound: 57.5125661
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -57.5087887, upper bound: 57.5087887
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -57.5138670, upper bound: 57.5427675
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -57.5087887, upper bound: 57.5392032
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -57.5226232, upper bound: 57.5141803
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -57.5392032, upper bound: 57.5192874
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -57.5226232, upper bound: 57.5280147
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -57.5392032, upper bound: 57.5497019
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -57.4927600, upper bound: 57.4817023
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -57.4932159, upper bound: 57.4818414
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -57.4927600, upper bound: 57.5432691
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -57.4932159, upper bound: 57.5422614
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -57.4926662, upper bound: 57.4803831
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -57.5130679, upper bound: 57.4875085
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -57.4926662, upper bound: 57.5079981
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -57.5130679, upper bound: 57.5497019
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -57.1475984, upper bound: 57.3365275
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -57.4818414, upper bound: 57.4932159
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -57.1475984, upper bound: 57.3669419
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -57.4818414, upper bound: 57.5130679
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -57.2967605, upper bound: 57.4165580
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -57.5422614, upper bound: 57.5202924
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -57.2967605, upper bound: 57.4338251
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -57.5422614, upper bound: 57.5533177
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -57.1322288, upper bound: 57.3096871
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -57.4665788, upper bound: 57.4665788
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -57.1322288, upper bound: 57.3669419
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -57.4665788, upper bound: 57.5130679
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -57.2555014, upper bound: 57.3391795
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -57.5240013, upper bound: 57.4916812
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -57.2555014, upper bound: 57.4176680
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 0, lower bound: -57.5240013, upper bound: 57.5620557

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -8.7669353, 38.1990242, -8.9245329, 37.9915428, -46.7584686, 47.1235542
1: -11.2745037, 43.2634735, -11.4030266, 43.0242538, -54.2987480, 54.6665001
2: -11.0006275, 42.8983345, -11.1846972, 42.6384811, -53.6391068, 54.0830307
3: -19.5009098, 46.0489769, -19.4911079, 45.8305550, -65.3314667, 65.5400848
4: -17.9422951, 44.1722069, -18.0231190, 43.9328003, -61.8750954, 62.1953239

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5087887, upper bound: 57.5087887
time: 0.52 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5087887, upper bound: 57.5087887
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8.8402529, 37.7630539, -9.1575928, 38.8744278, -47.7146759, 46.9206429
1: -11.2974596, 42.7681274, -11.6959438, 44.0212021, -55.3186607, 54.4640732
2: -11.0862007, 42.3602257, -11.4683819, 43.6604576, -54.7466583, 53.8286057
3: -19.3324356, 45.5596771, -19.9659348, 46.8748016, -66.2072372, 65.5256042
4: -17.8893318, 43.6359253, -18.4506874, 44.9865723, -62.8759041, 62.0866089

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5087887, upper bound: 57.5087887
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5087887, upper bound: 57.5087887
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -8.7669353, 38.1990242, -8.5814743, 36.4961777, -45.2631073, 46.7804985
1: -11.2745037, 43.2634735, -10.9528027, 41.3554649, -52.6299553, 54.2162781
2: -11.0006275, 42.8983345, -10.7885666, 40.8837357, -51.8843613, 53.6868973
3: -19.5009098, 46.0489769, -18.6880188, 44.1730118, -63.6739197, 64.7369919
4: -17.9422951, 44.1722069, -17.4680176, 42.1029663, -60.0452614, 61.6402245

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4978419, upper bound: 57.4877973
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5109099, upper bound: 57.5256938
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.8402529, 37.7630539, -8.8118200, 37.3577957, -46.1980476, 46.5748749
1: -11.2974596, 42.7681274, -11.2442751, 42.3229713, -53.6204300, 54.0124016
2: -11.0862007, 42.3602257, -11.0679474, 41.8824844, -52.9686852, 53.4281693
3: -19.3324356, 45.5596771, -19.1606712, 45.1869125, -64.5193405, 64.7203445
4: -17.8893318, 43.6359253, -17.8871288, 43.1344337, -61.0237617, 61.5230370

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5141803, upper bound: 57.5226232
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5141803, upper bound: 57.5392032
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -8.0646629, 35.2838440, -8.9245329, 37.9915428, -46.0562019, 44.2083778
1: -10.3480911, 39.9815979, -11.4030266, 43.0242538, -53.3723450, 51.3846245
2: -10.1572323, 39.4773331, -11.1846972, 42.6384811, -52.7957153, 50.6620293
3: -17.9082813, 42.6824608, -19.4911079, 45.8305550, -63.7388382, 62.1735687
4: -16.6487427, 40.6530380, -18.0231190, 43.9328003, -60.5815430, 58.6761551

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5226232, upper bound: 57.5141803
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5226232, upper bound: 57.5141803
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -8.4693804, 36.1834831, -9.1575928, 38.8744278, -47.3438072, 45.3410721
1: -10.8127460, 41.0070839, -11.6959438, 44.0212021, -54.8339462, 52.7030258
2: -10.6570034, 40.5084305, -11.4683819, 43.6604576, -54.3174591, 51.9768143
3: -18.4758568, 43.8035660, -19.9659348, 46.8748016, -65.3506622, 63.7694817
4: -17.2884102, 41.7039909, -18.4506874, 44.9865723, -62.2749825, 60.1546783

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5392032, upper bound: 57.5192874
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5392032, upper bound: 57.5192874
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -8.0646629, 35.2838440, -8.5814743, 36.4961777, -44.5608368, 43.8653183
1: -10.3480911, 39.9815979, -10.9528027, 41.3554649, -51.7035561, 50.9344025
2: -10.1572323, 39.4773331, -10.7885666, 40.8837357, -51.0409698, 50.2658958
3: -17.9082813, 42.6824608, -18.6880188, 44.1730118, -62.0812912, 61.3704796
4: -16.6487427, 40.6530380, -17.4680176, 42.1029663, -58.7517090, 58.1210556

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4962781, upper bound: 57.4639298
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5136952, upper bound: 57.5136952
time: 0.52 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -8.4693804, 36.1834831, -8.8118200, 37.3577957, -45.8271751, 44.9953041
1: -10.8127460, 41.0070839, -11.2442751, 42.3229713, -53.1357193, 52.2513580
2: -10.6570034, 40.5084305, -11.0679474, 41.8824844, -52.5394859, 51.5763702
3: -18.4758568, 43.8035660, -19.1606712, 45.1869125, -63.6627693, 62.9642296
4: -17.2884102, 41.7039909, -17.8871288, 43.1344337, -60.4228439, 59.5911179

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5445947, upper bound: 57.5331219
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5445947, upper bound: 57.5497019
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -8.7669353, 38.1990242, -10.8008080, 45.2789574, -54.0458832, 48.9998322
1: -11.2745037, 43.2634735, -13.7873459, 51.2305145, -62.5050125, 57.0508194
2: -11.0006275, 42.8983345, -13.4330664, 51.0943871, -62.0950012, 56.3313980
3: -19.5009098, 46.0489769, -23.3554573, 54.3990669, -73.8999786, 69.4044342
4: -17.9422951, 44.1722069, -21.4463768, 52.5510941, -70.4933929, 65.6185760

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3365275, upper bound: 57.1475984
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3365275, upper bound: 57.4817023
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8.8402529, 37.7630539, -11.0297852, 46.1416092, -54.9818611, 48.7928352
1: -11.2974596, 42.7681274, -14.0756989, 52.2037125, -63.5011711, 56.8438263
2: -11.0862007, 42.3602257, -13.7119274, 52.0922241, -63.1784248, 56.0721512
3: -19.3324356, 45.5596771, -23.8188839, 55.4197121, -74.7521515, 69.3785629
4: -17.8893318, 43.6359253, -21.8664837, 53.5777702, -71.4670792, 65.5024109

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3365275, upper bound: 57.1475984
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3365275, upper bound: 57.4818414
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -8.7669353, 38.1990242, -10.1803131, 42.6458778, -51.4128036, 48.3793373
1: -11.2745037, 43.2634735, -12.9598274, 48.2555504, -59.5300522, 56.2233009
2: -11.0006275, 42.8983345, -12.7167873, 47.9966087, -58.9972382, 55.6151199
3: -19.5009098, 46.0489769, -21.9676991, 51.3662262, -70.8671341, 68.0166702
4: -17.9422951, 44.1722069, -20.3960743, 49.3953743, -67.3376694, 64.5682831

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5235237, upper bound: 57.5432691
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5242891, upper bound: 57.5432691
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.8402529, 37.7630539, -10.4046993, 43.4960098, -52.3362617, 48.1677437
1: -11.2974596, 42.7681274, -13.2431650, 49.2144547, -60.5119095, 56.0112915
2: -11.0862007, 42.3602257, -12.9902172, 48.9800606, -60.0662613, 55.3504410
3: -19.3324356, 45.5596771, -22.4255047, 52.3740425, -71.7064743, 67.9851837
4: -17.8893318, 43.6359253, -20.8130035, 50.4081459, -68.2974548, 64.4489212

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4165580, upper bound: 57.2967605
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4165580, upper bound: 57.5422614
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -8.0646629, 35.2838440, -10.8008080, 45.2789574, -53.3436165, 46.0846519
1: -10.3480911, 39.9815979, -13.7873459, 51.2305145, -61.5786018, 53.7689438
2: -10.1572323, 39.4773331, -13.4330664, 51.0943871, -61.2516022, 52.9104004
3: -17.9082813, 42.6824608, -23.3554573, 54.3990669, -72.3073502, 66.0379181
4: -16.6487427, 40.6530380, -21.4463768, 52.5510941, -69.1998367, 62.0994148

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3503619, upper bound: 57.1529899
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3503619, upper bound: 57.4803831
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -8.4693804, 36.1834831, -11.0297852, 46.1416092, -54.6109848, 47.2132683
1: -10.8127460, 41.0070839, -14.0756989, 52.2037125, -63.0164566, 55.0827827
2: -10.6570034, 40.5084305, -13.7119274, 52.0922241, -62.7492294, 54.2203560
3: -18.4758568, 43.8035660, -23.8188839, 55.4197121, -73.8955688, 67.6224518
4: -17.2884102, 41.7039909, -21.8664837, 53.5777702, -70.8661804, 63.5704727

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3669419, upper bound: 57.1580971
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3669419, upper bound: 57.4875086
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -8.0646629, 35.2838440, -10.1803131, 42.6458778, -50.7105370, 45.4641571
1: -10.3480911, 39.9815979, -12.9598274, 48.2555504, -58.6036415, 52.9414253
2: -10.1572323, 39.4773331, -12.7167873, 47.9966087, -58.1538353, 52.1941223
3: -17.9082813, 42.6824608, -21.9676991, 51.3662262, -69.2745056, 64.6501617
4: -16.6487427, 40.6530380, -20.3960743, 49.3953743, -66.0441132, 61.0491104

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4134067, upper bound: 57.2699110
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4134067, upper bound: 57.5079981
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -8.4693804, 36.1834831, -10.4046993, 43.4960098, -51.9653816, 46.5881767
1: -10.8127460, 41.0070839, -13.2431650, 49.2144547, -60.0271988, 54.2502480
2: -10.6570034, 40.5084305, -12.9902172, 48.9800606, -59.6370621, 53.4986496
3: -18.4758568, 43.8035660, -22.4255047, 52.3740425, -70.8498993, 66.2290649
4: -17.2884102, 41.7039909, -20.8130035, 50.4081459, -67.6965485, 62.5169945

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4469725, upper bound: 57.3072592
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4469725, upper bound: 57.5497018
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -11.0847101, 47.3818054, -8.9245329, 37.9915428, -49.0762482, 56.3063316
1: -14.2560978, 53.6242714, -11.4030266, 43.0242538, -57.2803497, 65.0272980
2: -13.7948971, 53.5095329, -11.1846972, 42.6384811, -56.4333763, 64.6942291
3: -24.3539143, 56.8776131, -19.4911079, 45.8305550, -70.1844711, 76.3687210
4: -22.2534161, 54.9756432, -18.0231190, 43.9328003, -66.1862183, 72.9987640

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.1475984, upper bound: 57.3365275
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.1475984, upper bound: 57.3365275
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -10.6957531, 44.9465866, -9.1575928, 38.8744278, -49.5701714, 54.1041718
1: -13.6549959, 50.8562546, -11.6959438, 44.0212021, -57.6761971, 62.5522003
2: -13.3063459, 50.7025528, -11.4683819, 43.6604576, -56.9668045, 62.1709213
3: -23.1477661, 54.0001144, -19.9659348, 46.8748016, -70.0225677, 73.9660492
4: -21.2595654, 52.1365585, -18.4506874, 44.9865723, -66.2461395, 70.5872498

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4817023, upper bound: 57.4927600
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4817023, upper bound: 57.4932159
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -11.0847101, 47.3818054, -8.5814743, 36.4961777, -47.5808868, 55.9632759
1: -14.2560978, 53.6242714, -10.9528027, 41.3554649, -55.6115608, 64.5770645
2: -13.7948971, 53.5095329, -10.7885666, 40.8837357, -54.6786346, 64.2980957
3: -24.3539143, 56.8776131, -18.6880188, 44.1730118, -68.5269241, 75.5656281
4: -22.2534161, 54.9756432, -17.4680176, 42.1029663, -64.3563843, 72.4436569

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.1529899, upper bound: 57.3503619
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.1529899, upper bound: 57.3669419
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -10.6957531, 44.9465866, -8.8118200, 37.3577957, -48.0535431, 53.7584076
1: -13.6549959, 50.8562546, -11.2442751, 42.3229713, -55.9779625, 62.1005287
2: -13.3063459, 50.7025528, -11.0679474, 41.8824844, -55.1888313, 61.7704811
3: -23.1477661, 54.0001144, -19.1606712, 45.1869125, -68.3346786, 73.1607819
4: -21.2595654, 52.1365585, -17.8871288, 43.1344337, -64.3939972, 70.0236740

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4803831, upper bound: 57.4926662
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4803831, upper bound: 57.5130679
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -10.0193787, 42.8943863, -8.9245329, 37.9915428, -48.0109215, 51.8189125
1: -12.8247375, 48.5298729, -11.4030266, 43.0242538, -55.8489876, 59.9328995
2: -12.5208769, 48.2823448, -11.1846972, 42.6384811, -55.1593513, 59.4670410
3: -21.9548664, 51.6154327, -19.4911079, 45.8305550, -67.7854233, 71.1065369
4: -20.2769165, 49.6436996, -18.0231190, 43.9328003, -64.2097168, 67.6668167

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2967605, upper bound: 57.4165580
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2967605, upper bound: 57.4165580
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -10.0372343, 42.1927071, -9.1575928, 38.8744278, -48.9116554, 51.3502960
1: -12.7809610, 47.7437553, -11.6959438, 44.0212021, -56.8021622, 59.4396973
2: -12.5463152, 47.4627686, -11.4683819, 43.6604576, -56.2067719, 58.9311523
3: -21.6898041, 50.8244820, -19.9659348, 46.8748016, -68.5645981, 70.7904053
4: -20.1472874, 48.8353882, -18.4506874, 44.9865723, -65.1338577, 67.2860718

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5422614, upper bound: 57.5202924
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5422614, upper bound: 57.5202924
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -10.0193787, 42.8943863, -8.5814743, 36.4961777, -46.5155563, 51.4758606
1: -12.8247375, 48.5298729, -10.9528027, 41.3554649, -54.1802025, 59.4826736
2: -12.5208769, 48.2823448, -10.7885666, 40.8837357, -53.4046097, 59.0709076
3: -21.9548664, 51.6154327, -18.6880188, 44.1730118, -66.1278687, 70.3034515
4: -20.2769165, 49.6436996, -17.4680176, 42.1029663, -62.3798828, 67.1117172

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3021520, upper bound: 57.4303925
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3021520, upper bound: 57.4338251
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -10.0372343, 42.1927071, -8.8118200, 37.3577957, -47.3950272, 51.0045280
1: -12.7809610, 47.7437553, -11.2442751, 42.3229713, -55.1039314, 58.9880295
2: -12.5463152, 47.4627686, -11.0679474, 41.8824844, -54.4287987, 58.5307159
3: -21.6898041, 50.8244820, -19.1606712, 45.1869125, -66.8767090, 69.9851532
4: -20.1472874, 48.8353882, -17.8871288, 43.1344337, -63.2817154, 66.7224960

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5502469, upper bound: 57.5350720
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5502469, upper bound: 57.5533177
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -11.0847101, 47.3818054, -10.8008080, 45.2789574, -56.3636627, 58.1826134
1: -14.2560978, 53.6242714, -13.7873459, 51.2305145, -65.4866104, 67.4116058
2: -13.7948971, 53.5095329, -13.4330664, 51.0943871, -64.8892746, 66.9425964
3: -24.3539143, 56.8776131, -23.3554573, 54.3990669, -78.7529831, 80.2330704
4: -22.2534161, 54.9756432, -21.4463768, 52.5510941, -74.8045120, 76.4220123

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -56.9753371, upper bound: 56.9753371
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -56.9753371, upper bound: 57.3096871
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -10.6957531, 44.9465866, -11.0297852, 46.1416092, -56.8373566, 55.9763718
1: -13.6549959, 50.8562546, -14.0756989, 52.2037125, -65.8587036, 64.9319534
2: -13.3063459, 50.7025528, -13.7119274, 52.0922241, -65.3985672, 64.4144440
3: -23.1477661, 54.0001144, -23.8188839, 55.4197121, -78.5674744, 77.8190002
4: -21.2595654, 52.1365585, -21.8664837, 53.5777702, -74.8373337, 74.0030441

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3096871, upper bound: 57.1322288
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3096871, upper bound: 57.4665788
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -11.0847101, 47.3818054, -10.1803131, 42.6458778, -53.7305832, 57.5621147
1: -14.2560978, 53.6242714, -12.9598274, 48.2555504, -62.5116501, 66.5840988
2: -13.7948971, 53.5095329, -12.7167873, 47.9966087, -61.7914925, 66.2263184
3: -24.3539143, 56.8776131, -21.9676991, 51.3662262, -75.7201385, 78.8453140
4: -22.2534161, 54.9756432, -20.3960743, 49.3953743, -71.6487885, 75.3717194

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.1424428, upper bound: 57.3586280
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.1580971, upper bound: 57.3669419
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -10.6957531, 44.9465866, -10.4046993, 43.4960098, -54.1917572, 55.3512764
1: -13.6549959, 50.8562546, -13.2431650, 49.2144547, -62.8694344, 64.0994186
2: -13.3063459, 50.7025528, -12.9902172, 48.9800606, -62.2864075, 63.6927376
3: -23.1477661, 54.0001144, -22.4255047, 52.3740425, -75.5218048, 76.4256210
4: -21.2595654, 52.1365585, -20.8130035, 50.4081459, -71.6677094, 72.9495544

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3391795, upper bound: 57.2555014
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3391795, upper bound: 57.5130679
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -10.0193787, 42.8943863, -10.8008080, 45.2789574, -55.2983360, 53.6951942
1: -12.8247375, 48.5298729, -13.7873459, 51.2305145, -64.0552521, 62.3172188
2: -12.5208769, 48.2823448, -13.4330664, 51.0943871, -63.6152573, 61.7154121
3: -21.9548664, 51.6154327, -23.3554573, 54.3990669, -76.3539352, 74.9708862
4: -20.2769165, 49.6436996, -21.4463768, 52.5510941, -72.8280106, 71.0900726

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -57.1244992, upper bound: 57.0553677
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.1244992, upper bound: 57.3391795
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -10.0372343, 42.1927071, -11.0297852, 46.1416092, -56.1788368, 53.2224922
1: -12.7809610, 47.7437553, -14.0756989, 52.2037125, -64.9846725, 61.8194542
2: -12.5463152, 47.4627686, -13.7119274, 52.0922241, -64.6385422, 61.1746864
3: -21.6898041, 50.8244820, -23.8188839, 55.4197121, -77.1095123, 74.6433640
4: -20.1472874, 48.8353882, -21.8664837, 53.5777702, -73.7250519, 70.7018738

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3687549, upper bound: 57.1586122
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3687539, upper bound: 57.4916812
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -10.0193787, 42.8943863, -10.1803131, 42.6458778, -52.6652565, 53.0746956
1: -12.8247375, 48.5298729, -12.9598274, 48.2555504, -61.0802879, 61.4897003
2: -12.5208769, 48.2823448, -12.7167873, 47.9966087, -60.5174751, 60.9991264
3: -21.9548664, 51.6154327, -21.9676991, 51.3662262, -73.3210907, 73.5831299
4: -20.2769165, 49.6436996, -20.3960743, 49.3953743, -69.6722870, 70.0397720

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -57.2045298, upper bound: 57.2045298
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2045298, upper bound: 57.4176680
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -10.0372343, 42.1927071, -10.4046993, 43.4960098, -53.5332336, 52.5974007
1: -12.7809610, 47.7437553, -13.2431650, 49.2144547, -61.9954071, 60.9869194
2: -12.5463152, 47.4627686, -12.9902172, 48.9800606, -61.5263748, 60.4529800
3: -21.6898041, 50.8244820, -22.4255047, 52.3740425, -74.0638428, 73.2499847
4: -20.1472874, 48.8353882, -20.8130035, 50.4081459, -70.5554123, 69.6483841

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4527388, upper bound: 57.3092460
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4527388, upper bound: 57.5620557
time: 0.67 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.68 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.5087887, upper bound: 57.5087887
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.5087887, upper bound: 57.5087887
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.5087887, upper bound: 57.5087887
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.5087887, upper bound: 57.5087887
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.4978419, upper bound: 57.4877973
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.5109099, upper bound: 57.5256938
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.5141803, upper bound: 57.5226232
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.5141803, upper bound: 57.5392032
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.5226232, upper bound: 57.5141803
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.5226232, upper bound: 57.5141803
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.5392032, upper bound: 57.5192874
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.5392032, upper bound: 57.5192874
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.4962781, upper bound: 57.4639298
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.5136952, upper bound: 57.5136952
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.5445947, upper bound: 57.5331219
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.5445947, upper bound: 57.5497019
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.3365275, upper bound: 57.1475984
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.3365275, upper bound: 57.4817023
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.3365275, upper bound: 57.1475984
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.3365275, upper bound: 57.4818414
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.5235237, upper bound: 57.5432691
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.5242891, upper bound: 57.5432691
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.4165580, upper bound: 57.2967605
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.4165580, upper bound: 57.5422614
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.3503619, upper bound: 57.1529899
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.3503619, upper bound: 57.4803831
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.3669419, upper bound: 57.1580971
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.3669419, upper bound: 57.4875086
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.4134067, upper bound: 57.2699110
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.4134067, upper bound: 57.5079981
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.4469725, upper bound: 57.3072592
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.4469725, upper bound: 57.5497018
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.1475984, upper bound: 57.3365275
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.1475984, upper bound: 57.3365275
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.4817023, upper bound: 57.4927600
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.4817023, upper bound: 57.4932159
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.1529899, upper bound: 57.3503619
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.1529899, upper bound: 57.3669419
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.4803831, upper bound: 57.4926662
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.4803831, upper bound: 57.5130679
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.2967605, upper bound: 57.4165580
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.2967605, upper bound: 57.4165580
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.5422614, upper bound: 57.5202924
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.5422614, upper bound: 57.5202924
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.3021520, upper bound: 57.4303925
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.3021520, upper bound: 57.4338251
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.5502469, upper bound: 57.5350720
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.5502469, upper bound: 57.5533177
IS_A2_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.68
Output dim: 0, lower bound: -56.9753371, upper bound: 56.9753371
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -56.9753371, upper bound: 57.3096871
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.3096871, upper bound: 57.1322288
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.3096871, upper bound: 57.4665788
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.1424428, upper bound: 57.3586280
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.1580971, upper bound: 57.3669419
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.3391795, upper bound: 57.2555014
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.3391795, upper bound: 57.5130679
IS_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.1244992, upper bound: 57.0553677
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.1244992, upper bound: 57.3391795
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.3687549, upper bound: 57.1586122
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.3687539, upper bound: 57.4916812
IS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.2045298, upper bound: 57.2045298
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.2045298, upper bound: 57.4176680
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.4527388, upper bound: 57.3092460
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.68
Output dim: 0, lower bound: -57.4527388, upper bound: 57.5620557

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -8.7669353, 38.1990242, -8.7669353, 38.1990242, -46.9659538, 46.9659538
1: -11.2745037, 43.2634735, -11.2745037, 43.2634735, -54.5379753, 54.5379753
2: -11.0006275, 42.8983345, -11.0006275, 42.8983345, -53.8989639, 53.8989639
3: -19.5009098, 46.0489769, -19.5009098, 46.0489769, -65.5498886, 65.5498886
4: -17.9422951, 44.1722069, -17.9422951, 44.1722069, -62.1145020, 62.1145020

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4444451, upper bound: 57.4783391
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5011117, upper bound: 57.4986064
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -8.7669353, 38.1990242, -8.8402529, 37.7630539, -46.5299797, 47.0392761
1: -11.2745037, 43.2634735, -11.2974596, 42.7681274, -54.0426254, 54.5609322
2: -11.0006275, 42.8983345, -11.0862007, 42.3602257, -53.3608551, 53.9845352
3: -19.5009098, 46.0489769, -19.3324356, 45.5596771, -65.0605850, 65.3814087
4: -17.9422951, 44.1722069, -17.8893318, 43.6359253, -61.5782166, 62.0615311

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4444451, upper bound: 57.4783391
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5011117, upper bound: 57.4986064
time: 0.54 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.8402529, 37.7630539, -8.7669353, 38.1990242, -47.0392761, 46.5299797
1: -11.2974596, 42.7681274, -11.2745037, 43.2634735, -54.5609322, 54.0426254
2: -11.0862007, 42.3602257, -11.0006275, 42.8983345, -53.9845352, 53.3608551
3: -19.3324356, 45.5596771, -19.5009098, 46.0489769, -65.3814087, 65.0605850
4: -17.8893318, 43.6359253, -17.9422951, 44.1722069, -62.0615234, 61.5782166

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4364782, upper bound: 57.4738660
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4947912, upper bound: 57.4947913
time: 0.52 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.8402529, 37.7630539, -8.8402529, 37.7630539, -46.6033058, 46.6033058
1: -11.2974596, 42.7681274, -11.2974596, 42.7681274, -54.0655861, 54.0655861
2: -11.0862007, 42.3602257, -11.0862007, 42.3602257, -53.4464264, 53.4464264
3: -19.3324356, 45.5596771, -19.3324356, 45.5596771, -64.8921127, 64.8921127
4: -17.8893318, 43.6359253, -17.8893318, 43.6359253, -61.5252380, 61.5252419

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4364782, upper bound: 57.4738660
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4947912, upper bound: 57.4947913
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -8.7669353, 38.1990242, -5.6921964, 26.5544319, -35.3213654, 43.8912201
1: -11.2745037, 43.2634735, -7.2286553, 30.2651787, -41.5396729, 50.4921303
2: -11.0006275, 42.8983345, -7.3373003, 29.3258247, -40.3264503, 50.2356300
3: -19.5009098, 46.0489769, -12.7406349, 32.4852829, -51.9861908, 58.7896004
4: -17.9422951, 44.1722069, -12.3471918, 30.0169640, -47.9592590, 56.5193977

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4411753, upper bound: 57.4675301
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4411753, upper bound: 57.4877973
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -8.5379391, 37.3366165, -7.1254840, 31.2719917, -39.8099289, 44.4621010
1: -10.9822378, 42.2918663, -9.1104317, 35.5538445, -46.5360756, 51.4022980
2: -10.7246523, 41.9026794, -9.0617800, 34.7848053, -45.5094528, 50.9644585
3: -19.0220451, 45.0300179, -15.6471558, 38.1785889, -57.2006340, 60.6771660
4: -17.5168133, 43.1497498, -14.8086662, 35.7851868, -53.3019981, 57.9584122

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4540977, upper bound: 57.5054266
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4540977, upper bound: 57.5256938
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.8402529, 37.7630539, -8.0646629, 35.2838440, -44.1240959, 45.8277092
1: -11.2974596, 42.7681274, -10.3480911, 39.9815979, -51.2790565, 53.1162186
2: -11.0862007, 42.3602257, -10.1572323, 39.4773331, -50.5635338, 52.5174561
3: -19.3324356, 45.5596771, -17.9082813, 42.6824608, -62.0148964, 63.4679565
4: -17.8893318, 43.6359253, -16.6487427, 40.6530380, -58.5423698, 60.2846603

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4417602, upper bound: 57.4874880
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5000732, upper bound: 57.5084133
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.8402529, 37.7630539, -8.4693804, 36.1834831, -45.0237350, 46.2324257
1: -11.2974596, 42.7681274, -10.8127460, 41.0070839, -52.3045425, 53.5808716
2: -11.0862007, 42.3602257, -10.6570034, 40.5084305, -51.5946312, 53.0172272
3: -19.3324356, 45.5596771, -18.4758568, 43.8035660, -63.1359978, 64.0355377
4: -17.8893318, 43.6359253, -17.2884102, 41.7039909, -59.5933228, 60.9243355

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4417602, upper bound: 57.4927951
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5000732, upper bound: 57.5137939
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -8.0646629, 35.2838440, -8.7669353, 38.1990242, -46.2636833, 44.0507774
1: -10.3480911, 39.9815979, -11.2745037, 43.2634735, -53.6115646, 51.2560921
2: -10.1572323, 39.4773331, -11.0006275, 42.8983345, -53.0555649, 50.4779587
3: -17.9082813, 42.6824608, -19.5009098, 46.0489769, -63.9572601, 62.1833725
4: -16.6487427, 40.6530380, -17.9422951, 44.1722069, -60.8209457, 58.5953331

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4604355, upper bound: 57.4833397
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5084132, upper bound: 57.5000732
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -8.0646629, 35.2838440, -8.8402529, 37.7630539, -45.8277092, 44.1240959
1: -10.3480911, 39.9815979, -11.2974596, 42.7681274, -53.1162186, 51.2790565
2: -10.1572323, 39.4773331, -11.0862007, 42.3602257, -52.5174561, 50.5635338
3: -17.9082813, 42.6824608, -19.3324356, 45.5596771, -63.4679565, 62.0148964
4: -16.6487427, 40.6530380, -17.8893318, 43.6359253, -60.2846603, 58.5423698

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4604355, upper bound: 57.4833397
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5084132, upper bound: 57.5000732
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.4693804, 36.1834831, -8.7669353, 38.1990242, -46.6684036, 44.9504089
1: -10.8127460, 41.0070839, -11.2745037, 43.2634735, -54.0762177, 52.2815781
2: -10.6570034, 40.5084305, -11.0006275, 42.8983345, -53.5553360, 51.5090561
3: -18.4758568, 43.8035660, -19.5009098, 46.0489769, -64.5248337, 63.3044739
4: -17.2884102, 41.7039909, -17.9422951, 44.1722069, -61.4606171, 59.6462860

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4839822, upper bound: 57.4915215
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5218787, upper bound: 57.5045986
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.4693804, 36.1834831, -8.8402529, 37.7630539, -46.2324257, 45.0237350
1: -10.8127460, 41.0070839, -11.2974596, 42.7681274, -53.5808716, 52.3045425
2: -10.6570034, 40.5084305, -11.0862007, 42.3602257, -53.0172272, 51.5946312
3: -18.4758568, 43.8035660, -19.3324356, 45.5596771, -64.0355377, 63.1359978
4: -17.2884102, 41.7039909, -17.8893318, 43.6359253, -60.9243240, 59.5933228

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4839822, upper bound: 57.4915215
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5218787, upper bound: 57.5045986
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -8.0646629, 35.2838440, -5.6921964, 26.5544319, -34.6190948, 40.9760399
1: -10.3480911, 39.9815979, -7.2286553, 30.2651787, -40.6132698, 47.2102547
2: -10.1572323, 39.4773331, -7.3373003, 29.3258247, -39.4830551, 46.8146324
3: -17.9082813, 42.6824608, -12.7406349, 32.4852829, -50.3935623, 55.4230919
4: -16.6487427, 40.6530380, -12.3471918, 30.0169640, -46.6657028, 53.0002289

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4489839, upper bound: 57.4489839
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4489839, upper bound: 57.4489839
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -7.8685250, 34.5492744, -7.1254840, 31.2719917, -39.1405182, 41.6747589
1: -10.0962276, 39.1567459, -9.1104317, 35.5538445, -45.6500702, 48.2671776
2: -9.9210196, 38.6295242, -9.0617800, 34.7848053, -44.7058220, 47.6913033
3: -17.4936943, 41.8185501, -15.6471558, 38.1785889, -55.6722717, 57.4657059
4: -16.2863960, 39.7808685, -14.8086662, 35.7851868, -52.0715790, 54.5895348

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4659216, upper bound: 57.4976213
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4659216, upper bound: 57.5136952
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.4693804, 36.1834831, -8.0646629, 35.2838440, -43.7532234, 44.2481422
1: -10.8127460, 41.0070839, -10.3480911, 39.9815979, -50.7943420, 51.3551750
2: -10.6570034, 40.5084305, -10.1572323, 39.4773331, -50.1343384, 50.6656647
3: -18.4758568, 43.8035660, -17.9082813, 42.6824608, -61.1583176, 61.7118454
4: -17.2884102, 41.7039909, -16.6487427, 40.6530380, -57.9414482, 58.3527298

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4892642, upper bound: 57.5051435
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5271606, upper bound: 57.5182206
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.4693804, 36.1834831, -8.4693804, 36.1834831, -44.6528587, 44.6528587
1: -10.8127460, 41.0070839, -10.8127460, 41.0070839, -51.8198318, 51.8198318
2: -10.6570034, 40.5084305, -10.6570034, 40.5084305, -51.1654358, 51.1654320
3: -18.4758568, 43.8035660, -18.4758568, 43.8035660, -62.2794228, 62.2794228
4: -17.2884102, 41.7039909, -17.2884102, 41.7039909, -58.9924011, 58.9924011

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4892642, upper bound: 57.5168833
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5271606, upper bound: 57.5316335
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -8.7669353, 38.1990242, -11.0243721, 47.1488724, -55.9158020, 49.2233963
1: -11.2745037, 43.2634735, -14.1803865, 53.3598557, -64.6343460, 57.4438591
2: -11.0006275, 42.8983345, -13.7203350, 53.2445908, -64.2452087, 56.6186676
3: -19.5009098, 46.0489769, -24.2248249, 56.5950623, -76.0959549, 70.2737961
4: -17.9422951, 44.1722069, -22.1343040, 54.7041283, -72.6464157, 66.3065033

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3414447, upper bound: 57.1513678
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -57.2725323, upper bound: 57.1238339
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -8.7669353, 38.1990242, -10.6957531, 44.9465866, -53.7135086, 48.8947754
1: -11.2745037, 43.2634735, -13.6549959, 50.8562546, -62.1307487, 56.9184685
2: -11.0006275, 42.8983345, -13.3063459, 50.7025528, -61.7031593, 56.2046814
3: -19.5009098, 46.0489769, -23.1477661, 54.0001144, -73.5010223, 69.1967468
4: -17.9422951, 44.1722069, -21.2595654, 52.1365585, -70.0788574, 65.4317703

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3414447, upper bound: 57.4651380
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2725323, upper bound: 57.4419118
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.8402529, 37.7630539, -11.0847101, 47.3818054, -56.2220535, 48.8477554
1: -11.2974596, 42.7681274, -14.2560978, 53.6242714, -64.9217224, 57.0242233
2: -11.0862007, 42.3602257, -13.7948971, 53.5095329, -64.5957336, 56.1551208
3: -19.3324356, 45.5596771, -24.3539143, 56.8776131, -76.2100525, 69.9135895
4: -17.8893318, 43.6359253, -22.2534161, 54.9756432, -72.8649521, 65.8893433

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3230933, upper bound: 57.1390391
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3340600, upper bound: 57.1448730
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.8402529, 37.7630539, -10.6957531, 44.9465866, -53.7868385, 48.4588013
1: -11.2974596, 42.7681274, -13.6549959, 50.8562546, -62.1537056, 56.4231224
2: -11.0862007, 42.3602257, -13.3063459, 50.7025528, -61.7887383, 55.6665726
3: -19.3324356, 45.5596771, -23.1477661, 54.0001144, -73.3325500, 68.7074432
4: -17.8893318, 43.6359253, -21.2595654, 52.1365585, -70.0258789, 64.8954926

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3230933, upper bound: 57.4602896
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3340600, upper bound: 57.4641336
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -8.6924419, 37.9307938, -10.2531567, 42.4256477, -51.1180840, 48.1839485
1: -11.1812963, 42.9626465, -12.9840136, 48.0806007, -59.2618980, 55.9466591
2: -10.9106703, 42.5888176, -12.7901754, 47.7312851, -58.6419411, 55.3789825
3: -19.3507977, 45.7298622, -21.9063301, 51.3100815, -70.6608658, 67.6361923
4: -17.8029327, 43.8548660, -20.7263222, 48.9778137, -66.7807465, 64.5811920

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4513718, upper bound: 57.5076205
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5099498, upper bound: 57.5287679
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -8.7669353, 38.1990242, -9.9106150, 41.6189232, -50.3858528, 48.1096382
1: -11.2745037, 43.2634735, -12.6183357, 47.0983925, -58.3728714, 55.8818092
2: -11.0006275, 42.8983345, -12.3899088, 46.8146935, -57.8153191, 55.2882385
3: -19.5009098, 46.0489769, -21.4097195, 50.1461906, -69.6471024, 67.4586868
4: -17.9422951, 44.1722069, -19.8844643, 48.1836433, -66.1259384, 64.0566711

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4541081, upper bound: 57.5080670
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5110001, upper bound: 57.5289274
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.8402529, 37.7630539, -10.0193787, 42.8943863, -51.7346382, 47.7824326
1: -11.2974596, 42.7681274, -12.8247375, 48.5298729, -59.8273315, 55.5928650
2: -11.0862007, 42.3602257, -12.5208769, 48.2823448, -59.3685455, 54.8810959
3: -19.3324356, 45.5596771, -21.9548664, 51.6154327, -70.9478683, 67.5145416
4: -17.8893318, 43.6359253, -20.2769165, 49.6436996, -67.5330276, 63.9128418

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4035825, upper bound: 57.2887303
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4145493, upper bound: 57.2945642
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.8402529, 37.7630539, -10.0372343, 42.1927071, -51.0329590, 47.8002815
1: -11.2974596, 42.7681274, -12.7809610, 47.7437553, -59.0412064, 55.5490875
2: -11.0862007, 42.3602257, -12.5463152, 47.4627686, -58.5489693, 54.9065399
3: -19.3324356, 45.5596771, -21.6898041, 50.8244820, -70.1569214, 67.2494736
4: -17.8893318, 43.6359253, -20.1472874, 48.8353882, -66.7247009, 63.7832031

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4035825, upper bound: 57.5267281
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4145493, upper bound: 57.5311576
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -8.0646629, 35.2838440, -11.0243721, 47.1488724, -55.2135353, 46.3082123
1: -10.3480911, 39.9815979, -14.1803865, 53.3598557, -63.7079315, 54.1619835
2: -10.1572323, 39.4773331, -13.7203350, 53.2445908, -63.4018021, 53.1976700
3: -17.9082813, 42.6824608, -24.2248249, 56.5950623, -74.5033188, 66.9072876
4: -16.6487427, 40.6530380, -22.1343040, 54.7041283, -71.3528748, 62.7873344

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3405686, upper bound: 57.1478612
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3503619, upper bound: 57.1529899
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -8.0646629, 35.2838440, -10.6957531, 44.9465866, -53.0112457, 45.9795952
1: -10.3480911, 39.9815979, -13.6549959, 50.8562546, -61.2043457, 53.6365891
2: -10.1572323, 39.4773331, -13.3063459, 50.7025528, -60.8597527, 52.7836800
3: -17.9082813, 42.6824608, -23.1477661, 54.0001144, -71.9083939, 65.8302307
4: -16.6487427, 40.6530380, -21.2595654, 52.1365585, -68.7852936, 61.9126053

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3405686, upper bound: 57.4285562
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3503619, upper bound: 57.4803831
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.4693804, 36.1834831, -11.0847101, 47.3818054, -55.8511772, 47.2681885
1: -10.8127460, 41.0070839, -14.2560978, 53.6242714, -64.4370117, 55.2631836
2: -10.6570034, 40.5084305, -13.7948971, 53.5095329, -64.1665344, 54.3033257
3: -18.4758568, 43.8035660, -24.3539143, 56.8776131, -75.3534698, 68.1574783
4: -17.2884102, 41.7039909, -22.2534161, 54.9756432, -72.2640533, 63.9574051

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3504521, upper bound: 57.1497623
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3646260, upper bound: 57.1554089
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.4693804, 36.1834831, -10.6957531, 44.9465866, -53.4159584, 46.8792305
1: -10.8127460, 41.0070839, -13.6549959, 50.8562546, -61.6689987, 54.6620789
2: -10.6570034, 40.5084305, -13.3063459, 50.7025528, -61.3595390, 53.8147736
3: -18.4758568, 43.8035660, -23.1477661, 54.0001144, -72.4759674, 66.9513321
4: -17.2884102, 41.7039909, -21.2595654, 52.1365585, -69.4249725, 62.9635544

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3504521, upper bound: 57.4616349
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3646260, upper bound: 57.4641336
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -8.0646629, 35.2838440, -10.0193787, 42.8943863, -50.9590492, 45.3032227
1: -10.3480911, 39.9815979, -12.8247375, 48.5298729, -58.8779640, 52.8063354
2: -10.1572323, 39.4773331, -12.5208769, 48.2823448, -58.4395676, 51.9982071
3: -17.9082813, 42.6824608, -21.9548664, 51.6154327, -69.5237122, 64.6373291
4: -16.6487427, 40.6530380, -20.2769165, 49.6436996, -66.2924423, 60.9299545

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4069375, upper bound: 57.2699110
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4134067, upper bound: 57.2698164
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -8.0646629, 35.2838440, -10.0372343, 42.1927071, -50.2573624, 45.3210716
1: -10.3480911, 39.9815979, -12.7809610, 47.7437553, -58.0918388, 52.7625580
2: -10.1572323, 39.4773331, -12.5463152, 47.4627686, -57.6200027, 52.0236473
3: -17.9082813, 42.6824608, -21.6898041, 50.8244820, -68.7327652, 64.3722687
4: -16.6487427, 40.6530380, -20.1472874, 48.8353882, -65.4841309, 60.8003235

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4069375, upper bound: 57.5042800
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4134067, upper bound: 57.5079981
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.4693804, 36.1834831, -10.0193787, 42.8943863, -51.3637657, 46.2028618
1: -10.8127460, 41.0070839, -12.8247375, 48.5298729, -59.3426208, 53.8318214
2: -10.6570034, 40.5084305, -12.5208769, 48.2823448, -58.9393463, 53.0293007
3: -18.4758568, 43.8035660, -21.9548664, 51.6154327, -70.0912933, 65.7584305
4: -17.2884102, 41.7039909, -20.2769165, 49.6436996, -66.9321136, 61.9809074

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4309413, upper bound: 57.2994534
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4451152, upper bound: 57.3051000
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.4693804, 36.1834831, -10.0372343, 42.1927071, -50.6620827, 46.2207108
1: -10.8127460, 41.0070839, -12.7809610, 47.7437553, -58.5564957, 53.7880440
2: -10.6570034, 40.5084305, -12.5463152, 47.4627686, -58.1197739, 53.0547447
3: -18.4758568, 43.8035660, -21.6898041, 50.8244820, -69.3003387, 65.4933624
4: -17.2884102, 41.7039909, -20.1472874, 48.8353882, -66.1237946, 61.8512802

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4309413, upper bound: 57.5433040
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4451152, upper bound: 57.5483705
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -11.0847101, 47.3818054, -8.7669353, 38.1990242, -49.2837334, 56.1487274
1: -14.2560978, 53.6242714, -11.2745037, 43.2634735, -57.5195694, 64.8987656
2: -13.7948971, 53.5095329, -11.0006275, 42.8983345, -56.6932297, 64.5101624
3: -24.3539143, 56.8776131, -19.5009098, 46.0489769, -70.4028931, 76.3785248
4: -22.2534161, 54.9756432, -17.9422951, 44.1722069, -66.4256210, 72.9179382

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -56.4696834, upper bound: 56.7746788
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.1456139, upper bound: 57.3357367
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -11.0847101, 47.3818054, -8.8402529, 37.7630539, -48.8477554, 56.2220535
1: -14.2560978, 53.6242714, -11.2974596, 42.7681274, -57.0242233, 64.9217224
2: -13.7948971, 53.5095329, -11.0862007, 42.3602257, -56.1551208, 64.5957260
3: -24.3539143, 56.8776131, -19.3324356, 45.5596771, -69.9135895, 76.2100525
4: -22.2534161, 54.9756432, -17.8893318, 43.6359253, -65.8893433, 72.8649521

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -56.4696834, upper bound: 56.7746788
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.1456139, upper bound: 57.3357367
time: 0.54 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -10.6957531, 44.9465866, -8.7669353, 38.1990242, -48.8947754, 53.7135124
1: -13.6549959, 50.8562546, -11.2745037, 43.2634735, -56.9184685, 62.1307526
2: -13.3063459, 50.7025528, -11.0006275, 42.8983345, -56.2046814, 61.7031593
3: -23.1477661, 54.0001144, -19.5009098, 46.0489769, -69.1967468, 73.5010223
4: -21.2595654, 52.1365585, -17.9422951, 44.1722069, -65.4317703, 70.0788498

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -56.8343240, upper bound: 57.0961849
time: 0.52 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4815138, upper bound: 57.4926407
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -10.6957531, 44.9465866, -8.8402529, 37.7630539, -48.4588013, 53.7868347
1: -13.6549959, 50.8562546, -11.2974596, 42.7681274, -56.4231224, 62.1537132
2: -13.3063459, 50.7025528, -11.0862007, 42.3602257, -55.6665726, 61.7887306
3: -23.1477661, 54.0001144, -19.3324356, 45.5596771, -68.7074432, 73.3325500
4: -21.2595654, 52.1365585, -17.8893318, 43.6359253, -64.8954926, 70.0258713

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -56.8343240, upper bound: 57.1115451
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4815138, upper bound: 57.4930986
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -11.0847101, 47.3818054, -8.0646629, 35.2838440, -46.3685532, 55.4464607
1: -14.2560978, 53.6242714, -10.3480911, 39.9815979, -54.2376938, 63.9723549
2: -13.7948971, 53.5095329, -10.1572323, 39.4773331, -53.2722282, 63.6667595
3: -24.3539143, 56.8776131, -17.9082813, 42.6824608, -67.0363770, 74.7858963
4: -22.2534161, 54.9756432, -16.6487427, 40.6530380, -62.9064560, 71.6243820

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -56.4766725, upper bound: 56.7921053
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.1510054, upper bound: 57.3495428
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -11.0847101, 47.3818054, -8.4693804, 36.1834831, -47.2681885, 55.8511772
1: -14.2560978, 53.6242714, -10.8127460, 41.0070839, -55.2631798, 64.4370193
2: -13.7948971, 53.5095329, -10.6570034, 40.5084305, -54.3033218, 64.1665344
3: -24.3539143, 56.8776131, -18.4758568, 43.8035660, -68.1574783, 75.3534698
4: -22.2534161, 54.9756432, -17.2884102, 41.7039909, -63.9574051, 72.2640533

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -56.4766725, upper bound: 56.8055067
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.1510054, upper bound: 57.3659881
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -10.6957531, 44.9465866, -8.0646629, 35.2838440, -45.9795990, 53.0112419
1: -13.6549959, 50.8562546, -10.3480911, 39.9815979, -53.6365929, 61.2043419
2: -13.3063459, 50.7025528, -10.1572323, 39.4773331, -52.7836800, 60.8597641
3: -23.1477661, 54.0001144, -17.9082813, 42.6824608, -65.8302307, 71.9083939
4: -21.2595654, 52.1365585, -16.6487427, 40.6530380, -61.9126053, 68.7853012

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -56.6977356, upper bound: 57.0129216
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4801887, upper bound: 57.4925345
time: 0.61 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.21 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.4444451, upper bound: 57.4783391
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.5011117, upper bound: 57.4986064
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.4444451, upper bound: 57.4783391
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.5011117, upper bound: 57.4986064
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.4364782, upper bound: 57.4738660
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.4947912, upper bound: 57.4947913
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.4364782, upper bound: 57.4738660
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.4947912, upper bound: 57.4947913
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.4411753, upper bound: 57.4675301
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.4411753, upper bound: 57.4877973
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.4540977, upper bound: 57.5054266
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.4540977, upper bound: 57.5256938
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.4417602, upper bound: 57.4874880
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.5000732, upper bound: 57.5084133
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.4417602, upper bound: 57.4927951
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.5000732, upper bound: 57.5137939
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.4604355, upper bound: 57.4833397
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.5084132, upper bound: 57.5000732
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.4604355, upper bound: 57.4833397
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.5084132, upper bound: 57.5000732
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.4839822, upper bound: 57.4915215
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.5218787, upper bound: 57.5045986
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.4839822, upper bound: 57.4915215
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.5218787, upper bound: 57.5045986
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.4489839, upper bound: 57.4489839
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.4489839, upper bound: 57.4489839
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.4659216, upper bound: 57.4976213
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.4659216, upper bound: 57.5136952
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.4892642, upper bound: 57.5051435
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.5271606, upper bound: 57.5182206
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.4892642, upper bound: 57.5168833
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.5271606, upper bound: 57.5316335
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.3414447, upper bound: 57.1513678
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.2725323, upper bound: 57.1238339
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.3414447, upper bound: 57.4651380
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.2725323, upper bound: 57.4419118
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.3230933, upper bound: 57.1390391
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.3340600, upper bound: 57.1448730
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.3230933, upper bound: 57.4602896
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.3340600, upper bound: 57.4641336
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.4513718, upper bound: 57.5076205
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.5099498, upper bound: 57.5287679
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.4541081, upper bound: 57.5080670
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.5110001, upper bound: 57.5289274
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.4035825, upper bound: 57.2887303
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.4145493, upper bound: 57.2945642
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.4035825, upper bound: 57.5267281
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.4145493, upper bound: 57.5311576
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.3405686, upper bound: 57.1478612
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.3503619, upper bound: 57.1529899
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.3405686, upper bound: 57.4285562
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.3503619, upper bound: 57.4803831
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.3504521, upper bound: 57.1497623
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.3646260, upper bound: 57.1554089
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.3504521, upper bound: 57.4616349
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.3646260, upper bound: 57.4641336
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.4069375, upper bound: 57.2699110
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.4134067, upper bound: 57.2698164
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.4069375, upper bound: 57.5042800
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.4134067, upper bound: 57.5079981
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.4309413, upper bound: 57.2994534
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.4451152, upper bound: 57.3051000
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.4309413, upper bound: 57.5433040
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.4451152, upper bound: 57.5483705
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.21
Output dim: 0, lower bound: -56.4696834, upper bound: 56.7746788
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.1456139, upper bound: 57.3357367
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.21
Output dim: 0, lower bound: -56.4696834, upper bound: 56.7746788
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.1456139, upper bound: 57.3357367
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.21
Output dim: 0, lower bound: -56.8343240, upper bound: 57.0961849
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.4815138, upper bound: 57.4926407
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.21
Output dim: 0, lower bound: -56.8343240, upper bound: 57.1115451
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.4815138, upper bound: 57.4930986
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.21
Output dim: 0, lower bound: -56.4766725, upper bound: 56.7921053
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.1510054, upper bound: 57.3495428
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.21
Output dim: 0, lower bound: -56.4766725, upper bound: 56.8055067
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.1510054, upper bound: 57.3659881
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.21
Output dim: 0, lower bound: -56.6977356, upper bound: 57.0129216
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 0, lower bound: -57.4801887, upper bound: 57.4925345
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -57.4803831, upper bound: 57.5130679
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -57.2967605, upper bound: 57.4165580
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -57.2967605, upper bound: 57.4165580
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -57.5422614, upper bound: 57.5202924
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -57.5422614, upper bound: 57.5202924
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -57.3021520, upper bound: 57.4303925
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -57.3021520, upper bound: 57.4338251
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -57.5502469, upper bound: 57.5350720
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -57.5502469, upper bound: 57.5533177
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -56.9753371, upper bound: 57.3096871
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -57.3096871, upper bound: 57.1322288
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -57.3096871, upper bound: 57.4665788
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -57.1424428, upper bound: 57.3586280
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -57.1580971, upper bound: 57.3669419
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -57.3391795, upper bound: 57.2555014
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -57.3391795, upper bound: 57.5130679
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -57.1244992, upper bound: 57.3391795
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -57.3687549, upper bound: 57.1586122
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -57.3687539, upper bound: 57.4916812
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -57.2045298, upper bound: 57.4176680
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -57.4527388, upper bound: 57.3092460
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -57.4527388, upper bound: 57.5620557
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=66.57380676269531
rel_dist={0: [-57.5687467976788, 57.5687467976788]}

## Binary search (step 1) starts
Candidate diff: 0.1250000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

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
time: 0.62 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.28 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.28
Output dim: 0, lower bound: -57.5535923, upper bound: 57.5580881
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.28
Output dim: 0, lower bound: -57.5535923, upper bound: 57.5621985

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -10.1770201, 42.4380188, -12.0890970, 49.7964020, -59.9734230, 54.5271149
1: -12.9492474, 48.0443954, -15.3444796, 56.3365631, -69.2858124, 63.3888741
2: -12.7204800, 47.7395439, -15.0430412, 56.2083435, -68.9288254, 62.7825813
3: -21.9290562, 51.2056084, -25.8391190, 59.8623886, -81.7914352, 77.0447235
4: -20.4464874, 49.1660233, -24.0019188, 57.9184074, -78.3648987, 73.1679230

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5512705, upper bound: 57.5512705
time: 0.56 seconds

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

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5580881, upper bound: 57.5535923
time: 0.54 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5580881, upper bound: 57.5621985
time: 0.65 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.37 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.37
Output dim: 0, lower bound: -57.5512705, upper bound: 57.5512705
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.37
Output dim: 0, lower bound: -57.5512705, upper bound: 57.5580881
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.37
Output dim: 0, lower bound: -57.5580881, upper bound: 57.5535923
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.37
Output dim: 0, lower bound: -57.5580881, upper bound: 57.5621985

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -10.1770201, 42.4380188, -10.1770201, 42.4380188, -52.6150398, 52.6150398
1: -12.9492474, 48.0443954, -12.9492474, 48.0443954, -60.9936447, 60.9936447
2: -12.7204800, 47.7395439, -12.7204800, 47.7395439, -60.4600182, 60.4600143
3: -21.9290562, 51.2056084, -21.9290562, 51.2056084, -73.1346588, 73.1346588
4: -20.4464874, 49.1660233, -20.4464874, 49.1660233, -69.6124954, 69.6124954

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5131977, upper bound: 57.5301616
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5512705, upper bound: 57.5512705
time: 0.55 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -10.1770201, 42.4380188, -11.9793425, 49.4708633, -59.6478844, 54.4173622
1: -12.9492474, 48.0443954, -15.2106190, 55.9559326, -68.9051743, 63.2550125
2: -12.7204800, 47.7395439, -14.9106045, 55.8369446, -68.5574265, 62.6501427
3: -21.9290562, 51.2056084, -25.6317139, 59.4555740, -81.3846283, 76.8373108
4: -20.4464874, 49.1660233, -23.7928352, 57.5261345, -77.9726028, 72.9588547

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5131977, upper bound: 57.5301616
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5512705, upper bound: 57.5580881
time: 0.52 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -11.9793425, 49.4708633, -10.1770201, 42.4380188, -54.4173622, 59.6478844
1: -15.2106190, 55.9559326, -12.9492474, 48.0443954, -63.2550125, 68.9051743
2: -14.9106045, 55.8369446, -12.7204800, 47.7395439, -62.6501465, 68.5574265
3: -25.6317139, 59.4555740, -21.9290562, 51.2056084, -76.8373108, 81.3846283
4: -23.7928352, 57.5261345, -20.4464874, 49.1660233, -72.9588547, 77.9725952

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4785281, upper bound: 57.4907953
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5580880, upper bound: 57.5535923
time: 0.64 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -11.9793425, 49.4708633, -11.9793425, 49.4708633, -61.4502068, 61.4502068
1: -15.2106190, 55.9559326, -15.2106190, 55.9559326, -71.1665497, 71.1665497
2: -14.9106045, 55.8369446, -14.9106045, 55.8369446, -70.7475510, 70.7475510
3: -25.6317139, 59.4555740, -25.6317139, 59.4555740, -85.0872879, 85.0872879
4: -23.7928352, 57.5261345, -23.7928352, 57.5261345, -81.3189545, 81.3189621

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

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
- Time for IS candidates: 3.63 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.63
Output dim: 0, lower bound: -57.5131977, upper bound: 57.5301616
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.63
Output dim: 0, lower bound: -57.5512705, upper bound: 57.5512705
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.63
Output dim: 0, lower bound: -57.5131977, upper bound: 57.5301616
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.63
Output dim: 0, lower bound: -57.5512705, upper bound: 57.5580881
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.63
Output dim: 0, lower bound: -57.4785281, upper bound: 57.4907953
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.63
Output dim: 0, lower bound: -57.5580880, upper bound: 57.5535923
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.63
Output dim: 0, lower bound: -57.4785281, upper bound: 57.4907953
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.63
Output dim: 0, lower bound: -57.4785281, upper bound: 57.4907953

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -9.1575928, 38.8744278, -9.5324383, 40.0096436, -49.1672325, 48.4068680
1: -11.6959438, 44.0212021, -12.1431637, 45.3053627, -57.0013046, 56.1643677
2: -11.4683819, 43.6604576, -11.9324455, 44.9406128, -56.4089966, 55.5929031
3: -19.9659348, 46.8748016, -20.6196671, 48.3145409, -68.2804718, 67.4944534
4: -18.4506874, 44.9865723, -19.2304115, 46.2580605, -64.7087479, 64.2169800

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5189899, upper bound: 57.5189899
time: 0.52 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5189899, upper bound: 57.5340187
time: 0.54 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8.8118200, 37.3577957, -10.1770201, 42.4380188, -51.2498398, 47.5348167
1: -11.2442751, 42.3229713, -12.9492474, 48.0443954, -59.2886696, 55.2722168
2: -11.0679474, 41.8824844, -12.7204800, 47.7395439, -58.8074799, 54.6029587
3: -19.1606712, 45.1869125, -21.9290562, 51.2056084, -70.3662796, 67.1159668
4: -17.8871288, 43.1344337, -20.4464874, 49.1660233, -67.0531387, 63.5809097

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

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
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -9.1575928, 38.8744278, -11.2561569, 46.7493057, -55.9068909, 50.1305809
1: -11.6959438, 44.0212021, -14.3058786, 52.8868065, -64.5827484, 58.3270798
2: -11.4683819, 43.6604576, -14.0253191, 52.7105789, -64.1789627, 57.6857758
3: -19.9659348, 46.8748016, -24.1612244, 56.2217560, -76.1876831, 71.0360184
4: -18.4506874, 44.9865723, -22.4216366, 54.2697601, -72.7204437, 67.4082108

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4827866, upper bound: 57.4761779
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4827866, upper bound: 57.5301616
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.8118200, 37.3577957, -11.9793425, 49.4708633, -58.2826843, 49.3371353
1: -11.2442751, 42.3229713, -15.2106190, 55.9559326, -67.2002029, 57.5335922
2: -11.0679474, 41.8824844, -14.9106045, 55.8369446, -66.9048920, 56.7930908
3: -19.1606712, 45.1869125, -25.6317139, 59.4555740, -78.6162415, 70.8186111
4: -17.8871288, 43.1344337, -23.7928352, 57.5261345, -75.4132385, 66.9272690

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4907953, upper bound: 57.4785281
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4907953, upper bound: 57.5580880
time: 0.98 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -11.0297852, 46.1416092, -9.5324383, 40.0096436, -51.0394287, 55.6740494
1: -14.0756989, 52.2037125, -12.1431637, 45.3053627, -59.3810616, 64.3468781
2: -13.7119274, 52.0922241, -11.9324455, 44.9406128, -58.6525383, 64.0246735
3: -23.8188839, 55.4197121, -20.6196671, 48.3145409, -72.1334229, 76.0393829
4: -21.8664837, 53.5777702, -19.2304115, 46.2580605, -68.1245422, 72.8081741

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4761779, upper bound: 57.4827866
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4761779, upper bound: 57.4907953
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -10.4046993, 43.4960098, -10.1770201, 42.4380188, -52.8427162, 53.6730309
1: -13.2431650, 49.2144547, -12.9492474, 48.0443954, -61.2875595, 62.1637039
2: -12.9902172, 48.9800606, -12.7204800, 47.7395439, -60.7297478, 61.7005386
3: -22.4255047, 52.3740425, -21.9290562, 51.2056084, -73.6311111, 74.3030853
4: -20.8130035, 50.4081459, -20.4464874, 49.1660233, -69.9790115, 70.8546066

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5301616, upper bound: 57.5131977
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5301616, upper bound: 57.5535923
time: 1.15 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -11.0297852, 46.1416092, -11.2561569, 46.7493057, -57.7790871, 57.3977623
1: -14.0756989, 52.2037125, -14.3058786, 52.8868065, -66.9625092, 66.5095901
2: -13.7119274, 52.0922241, -14.0253191, 52.7105789, -66.4224854, 66.1175461
3: -23.8188839, 55.4197121, -24.1612244, 56.2217560, -80.0406418, 79.5809326
4: -21.8664837, 53.5777702, -22.4216366, 54.2697601, -76.1362457, 75.9994049

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4669328, upper bound: 57.4669328
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4669328, upper bound: 57.4907953
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -10.4046993, 43.4960098, -11.9793425, 49.4708633, -59.8755646, 55.4753494
1: -13.2431650, 49.2144547, -15.2106190, 55.9559326, -69.1990967, 64.4250717
2: -12.9902172, 48.9800606, -14.9106045, 55.8369446, -68.8271637, 63.8906631
3: -22.4255047, 52.3740425, -25.6317139, 59.4555740, -81.8810806, 78.0057449
4: -20.8130035, 50.4081459, -23.7928352, 57.5261345, -78.3391266, 74.2009735

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5020090, upper bound: 57.4820486
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4907953, upper bound: 57.5621587
time: 0.59 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.50 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.50
Output dim: 0, lower bound: -57.5189899, upper bound: 57.5189899
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.50
Output dim: 0, lower bound: -57.5189899, upper bound: 57.5340187
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.50
Output dim: 0, lower bound: -57.5340187, upper bound: 57.5226460
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.50
Output dim: 0, lower bound: -57.5340187, upper bound: 57.5512705
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.50
Output dim: 0, lower bound: -57.4827866, upper bound: 57.4761779
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.50
Output dim: 0, lower bound: -57.4827866, upper bound: 57.5301616
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.50
Output dim: 0, lower bound: -57.4907953, upper bound: 57.4785281
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.50
Output dim: 0, lower bound: -57.4907953, upper bound: 57.5580880
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.50
Output dim: 0, lower bound: -57.4761779, upper bound: 57.4827866
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.50
Output dim: 0, lower bound: -57.4761779, upper bound: 57.4907953
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.50
Output dim: 0, lower bound: -57.5301616, upper bound: 57.5131977
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.50
Output dim: 0, lower bound: -57.5301616, upper bound: 57.5535923
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.50
Output dim: 0, lower bound: -57.4669328, upper bound: 57.4669328
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.50
Output dim: 0, lower bound: -57.4669328, upper bound: 57.4907953
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.50
Output dim: 0, lower bound: -57.5020090, upper bound: 57.4820486
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.50
Output dim: 0, lower bound: -57.4907953, upper bound: 57.5621587

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -9.1575928, 38.8744278, -9.1575928, 38.8744278, -48.0320168, 48.0320168
1: -11.6959438, 44.0212021, -11.6959438, 44.0212021, -55.7171478, 55.7171478
2: -11.4683819, 43.6604576, -11.4683819, 43.6604576, -55.1288376, 55.1288376
3: -19.9659348, 46.8748016, -19.9659348, 46.8748016, -66.8407288, 66.8407288
4: -18.4506874, 44.9865723, -18.4506874, 44.9865723, -63.4372597, 63.4372597

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5138670, upper bound: 57.5125661
time: 0.51 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5087887, upper bound: 57.5087887
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -9.1575928, 38.8744278, -8.8118200, 37.3577957, -46.5153885, 47.6862488
1: -11.6959438, 44.0212021, -11.2442751, 42.3229713, -54.0189133, 55.2654762
2: -11.4683819, 43.6604576, -11.0679474, 41.8824844, -53.3508682, 54.7284012
3: -19.9659348, 46.8748016, -19.1606712, 45.1869125, -65.1528397, 66.0354691
4: -18.4506874, 44.9865723, -17.8871288, 43.1344337, -61.5851212, 62.8737030

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5138670, upper bound: 57.5314667
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5087887, upper bound: 57.5315670
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.8118200, 37.3577957, -9.1575928, 38.8744278, -47.6862488, 46.5153885
1: -11.2442751, 42.3229713, -11.6959438, 44.0212021, -55.2654762, 54.0189133
2: -11.0679474, 41.8824844, -11.4683819, 43.6604576, -54.7284012, 53.3508682
3: -19.1606712, 45.1869125, -19.9659348, 46.8748016, -66.0354691, 65.1528397
4: -17.8871288, 43.1344337, -18.4506874, 44.9865723, -62.8737030, 61.5851212

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5226232, upper bound: 57.5141803
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5315670, upper bound: 57.5166199
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.8118200, 37.3577957, -8.8118200, 37.3577957, -46.1696167, 46.1696167
1: -11.2442751, 42.3229713, -11.2442751, 42.3229713, -53.5672455, 53.5672455
2: -11.0679474, 41.8824844, -11.0679474, 41.8824844, -52.9504280, 52.9504280
3: -19.1606712, 45.1869125, -19.1606712, 45.1869125, -64.3475800, 64.3475800
4: -17.8871288, 43.1344337, -17.8871288, 43.1344337, -61.0215607, 61.0215607

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5226232, upper bound: 57.5280147
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5315670, upper bound: 57.5497019
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.1575928, 38.8744278, -11.0297852, 46.1416092, -55.2991982, 49.9042091
1: -11.6959438, 44.0212021, -14.0756989, 52.2037125, -63.8996582, 58.0969009
2: -11.4683819, 43.6604576, -13.7119274, 52.0922241, -63.5606079, 57.3723793
3: -19.9659348, 46.8748016, -23.8188839, 55.4197121, -75.3856506, 70.6936874
4: -18.4506874, 44.9865723, -21.8664837, 53.5777702, -72.0284576, 66.8530579

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4563171, upper bound: 57.4404028
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4822205, upper bound: 57.4753854
time: 0.51 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.1575928, 38.8744278, -10.4046993, 43.4960098, -52.6535950, 49.2791214
1: -11.6959438, 44.0212021, -13.2431650, 49.2144547, -60.9104004, 57.2643661
2: -11.4683819, 43.6604576, -12.9902172, 48.9800606, -60.4484406, 56.6506729
3: -19.9659348, 46.8748016, -22.4255047, 52.3740425, -72.3399658, 69.3003082
4: -18.4506874, 44.9865723, -20.8130035, 50.4081459, -68.8588257, 65.7995758

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4563171, upper bound: 57.5135716
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4822205, upper bound: 57.5284482
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.8118200, 37.3577957, -11.0297852, 46.1416092, -54.9534302, 48.3875809
1: -11.2442751, 42.3229713, -14.0756989, 52.2037125, -63.4479866, 56.3986702
2: -11.0679474, 41.8824844, -13.7119274, 52.0922241, -63.1601715, 55.5944099
3: -19.1606712, 45.1869125, -23.8188839, 55.4197121, -74.5803833, 69.0057983
4: -17.8871288, 43.1344337, -21.8664837, 53.5777702, -71.4648895, 65.0009155

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4334922, upper bound: 57.3370890
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4904413, upper bound: 57.4781048
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.8118200, 37.3577957, -10.4046993, 43.4960098, -52.3078308, 47.7624931
1: -11.2442751, 42.3229713, -13.2431650, 49.2144547, -60.4587288, 55.5661354
2: -11.0679474, 41.8824844, -12.9902172, 48.9800606, -60.0480080, 54.8726997
3: -19.1606712, 45.1869125, -22.4255047, 52.3740425, -71.5347137, 67.6124191
4: -17.8871288, 43.1344337, -20.8130035, 50.4081459, -68.2952499, 63.9474373

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4334922, upper bound: 57.4244370
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4904413, upper bound: 57.5497018
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -11.0297852, 46.1416092, -9.1575928, 38.8744278, -49.9042130, 55.2991982
1: -14.0756989, 52.2037125, -11.6959438, 44.0212021, -58.0969009, 63.8996582
2: -13.7119274, 52.0922241, -11.4683819, 43.6604576, -57.3723831, 63.5606079
3: -23.8188839, 55.4197121, -19.9659348, 46.8748016, -70.6936874, 75.3856506
4: -21.8664837, 53.5777702, -18.4506874, 44.9865723, -66.8530579, 72.0284576

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.1357383, upper bound: 57.3036922
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4753854, upper bound: 57.4822205
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -11.0297852, 46.1416092, -8.8118200, 37.3577957, -48.3875809, 54.9534302
1: -14.0756989, 52.2037125, -11.2442751, 42.3229713, -56.3986702, 63.4479866
2: -13.7119274, 52.0922241, -11.0679474, 41.8824844, -55.5944099, 63.1601715
3: -23.8188839, 55.4197121, -19.1606712, 45.1869125, -69.0057983, 74.5803833
4: -21.8664837, 53.5777702, -17.8871288, 43.1344337, -65.0009155, 71.4648895

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.1357383, upper bound: 57.3271205
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4753854, upper bound: 57.4904413
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -10.4046993, 43.4960098, -9.1575928, 38.8744278, -49.2791176, 52.6535950
1: -13.2431650, 49.2144547, -11.6959438, 44.0212021, -57.2643661, 60.9104004
2: -12.9902172, 48.9800606, -11.4683819, 43.6604576, -56.6506729, 60.4484406
3: -22.4255047, 52.3740425, -19.9659348, 46.8748016, -69.3003006, 72.3399658
4: -20.8130035, 50.4081459, -18.4506874, 44.9865723, -65.7995758, 68.8588257

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2967605, upper bound: 57.4165580
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5284482, upper bound: 57.5112195
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -10.4046993, 43.4960098, -8.8118200, 37.3577957, -47.7624893, 52.3078308
1: -13.2431650, 49.2144547, -11.2442751, 42.3229713, -55.5661354, 60.4587288
2: -12.9902172, 48.9800606, -11.0679474, 41.8824844, -54.8726997, 60.0480080
3: -22.4255047, 52.3740425, -19.1606712, 45.1869125, -67.6124115, 71.5347061
4: -20.8130035, 50.4081459, -17.8871288, 43.1344337, -63.9474335, 68.2952499

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2967605, upper bound: 57.4338251
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5284482, upper bound: 57.5523846
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -11.0297852, 46.1416092, -11.0297852, 46.1416092, -57.1713943, 57.1713943
1: -14.0756989, 52.2037125, -14.0756989, 52.2037125, -66.2794113, 66.2794113
2: -13.7119274, 52.0922241, -13.7119274, 52.0922241, -65.8041458, 65.8041458
3: -23.8188839, 55.4197121, -23.8188839, 55.4197121, -79.2385941, 79.2385941
4: -21.8664837, 53.5777702, -21.8664837, 53.5777702, -75.4442520, 75.4442520

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -57.0956875, upper bound: 57.2165477
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4665788, upper bound: 57.4665788
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -11.0297852, 46.1416092, -10.4046993, 43.4960098, -54.5257912, 56.5463028
1: -14.0756989, 52.2037125, -13.2431650, 49.2144547, -63.2901535, 65.4468765
2: -13.7119274, 52.0922241, -12.9902172, 48.9800606, -62.6919861, 65.0824432
3: -23.8188839, 55.4197121, -22.4255047, 52.3740425, -76.1929245, 77.8452148
4: -21.8664837, 53.5777702, -20.8130035, 50.4081459, -72.2746277, 74.3907700

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.0956875, upper bound: 57.3156531
time: 0.56 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4665788, upper bound: 57.4904413
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -10.4046993, 43.4960098, -11.0297852, 46.1416092, -56.5463028, 54.5257950
1: -13.2431650, 49.2144547, -14.0756989, 52.2037125, -65.4468765, 63.2901535
2: -12.9902172, 48.9800606, -13.7119274, 52.0922241, -65.0824432, 62.6919861
3: -22.4255047, 52.3740425, -23.8188839, 55.4197121, -77.8452148, 76.1929245
4: -20.8130035, 50.4081459, -21.8664837, 53.5777702, -74.3907700, 72.2746277

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -57.1524313, upper bound: 57.2091961
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5016550, upper bound: 57.4817208
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -10.4046993, 43.4960098, -10.4046993, 43.4960098, -53.9007034, 53.9006996
1: -13.2431650, 49.2144547, -13.2431650, 49.2144547, -62.4576187, 62.4576187
2: -12.9902172, 48.9800606, -12.9902172, 48.9800606, -61.9702759, 61.9702644
3: -22.4255047, 52.3740425, -22.4255047, 52.3740425, -74.7995453, 74.7995453
4: -20.8130035, 50.4081459, -20.8130035, 50.4081459, -71.2211304, 71.2211304

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33

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
time: 0.90 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.84 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -57.5138670, upper bound: 57.5125661
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -57.5087887, upper bound: 57.5087887
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -57.5138670, upper bound: 57.5314667
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -57.5087887, upper bound: 57.5315670
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -57.5226232, upper bound: 57.5141803
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -57.5315670, upper bound: 57.5166199
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -57.5226232, upper bound: 57.5280147
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -57.5315670, upper bound: 57.5497019
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -57.4563171, upper bound: 57.4404028
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -57.4822205, upper bound: 57.4753854
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -57.4563171, upper bound: 57.5135716
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -57.4822205, upper bound: 57.5284482
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -57.4334922, upper bound: 57.3370890
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -57.4904413, upper bound: 57.4781048
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -57.4334922, upper bound: 57.4244370
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -57.4904413, upper bound: 57.5497018
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -57.1357383, upper bound: 57.3036922
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -57.4753854, upper bound: 57.4822205
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -57.1357383, upper bound: 57.3271205
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -57.4753854, upper bound: 57.4904413
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -57.2967605, upper bound: 57.4165580
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -57.5284482, upper bound: 57.5112195
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -57.2967605, upper bound: 57.4338251
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -57.5284482, upper bound: 57.5523846
IS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.84
Output dim: 0, lower bound: -57.0956875, upper bound: 57.2165477
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -57.4665788, upper bound: 57.4665788
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -57.0956875, upper bound: 57.3156531
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -57.4665788, upper bound: 57.4904413
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.84
Output dim: 0, lower bound: -57.1524313, upper bound: 57.2091961
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -57.5016550, upper bound: 57.4817208
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -57.1524313, upper bound: 57.3737716
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -57.5016550, upper bound: 57.5620557

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -8.7669353, 38.1990242, -8.3100739, 35.6519890, -44.4189186, 46.5090981
1: -11.2745037, 43.2634735, -10.6276932, 40.3892288, -51.6637230, 53.8911667
2: -11.0006275, 42.8983345, -10.4366865, 39.9314880, -50.9321136, 53.3350182
3: -19.5009098, 46.0489769, -18.2305756, 43.0719185, -62.5728302, 64.2795486
4: -17.9422951, 44.1722069, -16.8955002, 41.1396370, -59.0819321, 61.0677032

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4801864, upper bound: 57.4402933
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5011117, upper bound: 57.4986063
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8.8402529, 37.7630539, -9.1575928, 38.8744278, -47.7146759, 46.9206429
1: -11.2974596, 42.7681274, -11.6959438, 44.0212021, -55.3186607, 54.4640732
2: -11.0862007, 42.3602257, -11.4683819, 43.6604576, -54.7466583, 53.8286057
3: -19.3324356, 45.5596771, -19.9659348, 46.8748016, -66.2072372, 65.5256042
4: -17.8893318, 43.6359253, -18.4506874, 44.9865723, -62.8759041, 62.0866089

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5087887, upper bound: 57.5087887
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5087887, upper bound: 57.5087887
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -8.7669353, 38.1990242, -7.9867325, 34.2763634, -43.0432930, 46.1857529
1: -11.2745037, 43.2634735, -10.1965561, 38.8652573, -50.1397514, 53.4600296
2: -11.0006275, 42.8983345, -10.0668869, 38.3097916, -49.3104172, 52.9652214
3: -19.5009098, 46.0489769, -17.4640160, 41.5618553, -61.0627670, 63.5129929
4: -17.9422951, 44.1722069, -16.3867416, 39.4446716, -57.3869553, 60.5589485

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4921367, upper bound: 57.4743048
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5076615, upper bound: 57.5197585
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.8402529, 37.7630539, -8.8118200, 37.3577957, -46.1980476, 46.5748749
1: -11.2974596, 42.7681274, -11.2442751, 42.3229713, -53.6204300, 54.0124016
2: -11.0862007, 42.3602257, -11.0679474, 41.8824844, -52.9686852, 53.4281693
3: -19.3324356, 45.5596771, -19.1606712, 45.1869125, -64.5193405, 64.7203445
4: -17.8893318, 43.6359253, -17.8871288, 43.1344337, -61.0237617, 61.5230370

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5141803, upper bound: 57.5226232
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5141803, upper bound: 57.5315670
time: 1.20 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -8.0646629, 35.2838440, -8.3100739, 35.6519890, -43.7166481, 43.5939178
1: -10.3480911, 39.9815979, -10.6276932, 40.3892288, -50.7373199, 50.6092911
2: -10.1572323, 39.4773331, -10.4366865, 39.9314880, -50.0887222, 49.9140205
3: -17.9082813, 42.6824608, -18.2305756, 43.0719185, -60.9802017, 60.9130363
4: -16.6487427, 40.6530380, -16.8955002, 41.1396370, -57.7883797, 57.5485382

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4874880, upper bound: 57.4417602
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5084133, upper bound: 57.5000732
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -8.4693804, 36.1834831, -9.1575928, 38.8744278, -47.3438072, 45.3410721
1: -10.8127460, 41.0070839, -11.6959438, 44.0212021, -54.8339462, 52.7030258
2: -10.6570034, 40.5084305, -11.4683819, 43.6604576, -54.3174591, 51.9768143
3: -18.4758568, 43.8035660, -19.9659348, 46.8748016, -65.3506622, 63.7694817
4: -17.2884102, 41.7039909, -18.4506874, 44.9865723, -62.2749825, 60.1546783

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5313612, upper bound: 57.5166199
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5313612, upper bound: 57.5166199
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -8.0646629, 35.2838440, -7.9867325, 34.2763634, -42.3410225, 43.2705727
1: -10.3480911, 39.9815979, -10.1965561, 38.8652573, -49.2133484, 50.1781540
2: -10.1572323, 39.4773331, -10.0668869, 38.3097916, -48.4670258, 49.5442200
3: -17.9082813, 42.6824608, -17.4640160, 41.5618553, -59.4701385, 60.1464767
4: -16.6487427, 40.6530380, -16.3867416, 39.4446716, -56.0934067, 57.0397797

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4962781, upper bound: 57.4639298
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5136952, upper bound: 57.5136952
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -8.4693804, 36.1834831, -8.8118200, 37.3577957, -45.8271751, 44.9953041
1: -10.8127460, 41.0070839, -11.2442751, 42.3229713, -53.1357193, 52.2513580
2: -10.6570034, 40.5084305, -11.0679474, 41.8824844, -52.5394859, 51.5763702
3: -18.4758568, 43.8035660, -19.1606712, 45.1869125, -63.6627693, 62.9642296
4: -17.2884102, 41.7039909, -17.8871288, 43.1344337, -60.4228439, 59.5911179

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5411622, upper bound: 57.5318641
time: 0.53 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5411622, upper bound: 57.5497019
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -8.7669353, 38.1990242, -10.2076769, 43.0309334, -51.7978630, 48.4067001
1: -11.2745037, 43.2634735, -13.0388489, 48.6942406, -59.9687386, 56.3023224
2: -11.0006275, 42.8983345, -12.7117128, 48.4935493, -59.4941788, 55.6100464
3: -19.5009098, 46.0489769, -22.1507416, 51.7412720, -71.2421799, 68.1997223
4: -17.9422951, 44.1722069, -20.3606224, 49.8761826, -67.8184814, 64.5328140

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3036922, upper bound: 57.1357383
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3036922, upper bound: 57.4404028
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8.8402529, 37.7630539, -11.0297852, 46.1416092, -54.9818611, 48.7928352
1: -11.2974596, 42.7681274, -14.0756989, 52.2037125, -63.5011711, 56.8438263
2: -11.0862007, 42.3602257, -13.7119274, 52.0922241, -63.1784248, 56.0721512
3: -19.3324356, 45.5596771, -23.8188839, 55.4197121, -74.7521515, 69.3785629
4: -17.8893318, 43.6359253, -21.8664837, 53.5777702, -71.4670792, 65.5024109

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3036922, upper bound: 57.1357383
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3036922, upper bound: 57.4753854
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -8.7669353, 38.1990242, -9.6080275, 40.5105820, -49.2775116, 47.8070526
1: -11.2745037, 43.2634735, -12.2365494, 45.8477631, -57.1222572, 55.5000229
2: -11.0006275, 42.8983345, -12.0228996, 45.5226021, -56.5232315, 54.9212303
3: -19.5009098, 46.0489769, -20.8062382, 48.8360825, -68.3369904, 66.8552170
4: -17.9422951, 44.1722069, -19.3448238, 46.8458099, -64.7880936, 63.5170135

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4815300, upper bound: 57.4574505
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4947398, upper bound: 57.5011140
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.8402529, 37.7630539, -10.4046993, 43.4960098, -52.3362617, 48.1677437
1: -11.2974596, 42.7681274, -13.2431650, 49.2144547, -60.5119095, 56.0112915
2: -11.0862007, 42.3602257, -12.9902172, 48.9800606, -60.0662613, 55.3504410
3: -19.3324356, 45.5596771, -22.4255047, 52.3740425, -71.7064743, 67.9851837
4: -17.8893318, 43.6359253, -20.8130035, 50.4081459, -68.2974548, 64.4489212

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4165580, upper bound: 57.2967605
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4165580, upper bound: 57.5284482
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -8.0646629, 35.2838440, -10.2076769, 43.0309334, -51.0955925, 45.4915199
1: -10.3480911, 39.9815979, -13.0388489, 48.6942406, -59.0423317, 53.0204468
2: -10.1572323, 39.4773331, -12.7117128, 48.4935493, -58.6507759, 52.1890450
3: -17.9082813, 42.6824608, -22.1507416, 51.7412720, -69.6495514, 64.8332062
4: -16.6487427, 40.6530380, -20.3606224, 49.8761826, -66.5249252, 61.0136490

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3201454, upper bound: 57.1423158
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3201454, upper bound: 57.3370890
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -8.4693804, 36.1834831, -11.0297852, 46.1416092, -54.6109848, 47.2132683
1: -10.8127460, 41.0070839, -14.0756989, 52.2037125, -63.0164566, 55.0827827
2: -10.6570034, 40.5084305, -13.7119274, 52.0922241, -62.7492294, 54.2203560
3: -18.4758568, 43.8035660, -23.8188839, 55.4197121, -73.8955688, 67.6224518
4: -17.2884102, 41.7039909, -21.8664837, 53.5777702, -70.8661804, 63.5704727

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3271205, upper bound: 57.1443533
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3271205, upper bound: 57.4781048
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -8.0646629, 35.2838440, -9.6080275, 40.5105820, -48.5752411, 44.8918724
1: -10.3480911, 39.9815979, -12.2365494, 45.8477631, -56.1958542, 52.2181435
2: -10.1572323, 39.4773331, -12.0228996, 45.5226021, -55.6798325, 51.5002327
3: -17.9082813, 42.6824608, -20.8062382, 48.8360825, -66.7443542, 63.4887009
4: -16.6487427, 40.6530380, -19.3448238, 46.8458099, -63.4945374, 59.9978485

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4670644, upper bound: 57.3944944
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4081399, upper bound: 57.3521439
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -8.4693804, 36.1834831, -10.4046993, 43.4960098, -51.9653816, 46.5881767
1: -10.8127460, 41.0070839, -13.2431650, 49.2144547, -60.0271988, 54.2502480
2: -10.6570034, 40.5084305, -12.9902172, 48.9800606, -59.6370621, 53.4986496
3: -18.4758568, 43.8035660, -22.4255047, 52.3740425, -70.8498993, 66.2290649
4: -17.2884102, 41.7039909, -20.8130035, 50.4081459, -67.6965485, 62.5169945

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4437227, upper bound: 57.3060635
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4437227, upper bound: 57.5497017
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -11.0847101, 47.3818054, -8.3100739, 35.6519890, -46.7366943, 55.6918716
1: -14.2560978, 53.6242714, -10.6276932, 40.3892288, -54.6453247, 64.2519684
2: -13.7948971, 53.5095329, -10.4366865, 39.9314880, -53.7263832, 63.9462204
3: -24.3539143, 56.8776131, -18.2305756, 43.0719185, -67.4258347, 75.1081848
4: -22.2534161, 54.9756432, -16.8955002, 41.1396370, -63.3930511, 71.8711319

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.1357383, upper bound: 57.3036922
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.1357383, upper bound: 57.3036922
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -10.6957531, 44.9465866, -9.1575928, 38.8744278, -49.5701714, 54.1041718
1: -13.6549959, 50.8562546, -11.6959438, 44.0212021, -57.6761971, 62.5522003
2: -13.3063459, 50.7025528, -11.4683819, 43.6604576, -56.9668045, 62.1709213
3: -23.1477661, 54.0001144, -19.9659348, 46.8748016, -70.0225677, 73.9660492
4: -21.2595654, 52.1365585, -18.4506874, 44.9865723, -66.2461395, 70.5872498

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4404028, upper bound: 57.4563171
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4404028, upper bound: 57.4822205
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -11.0847101, 47.3818054, -7.9867325, 34.2763634, -45.3610725, 55.3685265
1: -14.2560978, 53.6242714, -10.1965561, 38.8652573, -53.1213531, 63.8208199
2: -13.7948971, 53.5095329, -10.0668869, 38.3097916, -52.1046867, 63.5764198
3: -24.3539143, 56.8776131, -17.4640160, 41.5618553, -65.9157715, 74.3416290
4: -22.2534161, 54.9756432, -16.3867416, 39.4446716, -61.6980896, 71.3623810

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.1423158, upper bound: 57.3201454
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.1423158, upper bound: 57.3271205
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -10.6957531, 44.9465866, -8.8118200, 37.3577957, -48.0535431, 53.7584076
1: -13.6549959, 50.8562546, -11.2442751, 42.3229713, -55.9779625, 62.1005287
2: -13.3063459, 50.7025528, -11.0679474, 41.8824844, -55.1888313, 61.7704811
3: -23.1477661, 54.0001144, -19.1606712, 45.1869125, -68.3346786, 73.1607819
4: -21.2595654, 52.1365585, -17.8871288, 43.1344337, -64.3939972, 70.0236740

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3370890, upper bound: 57.4334922
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3370890, upper bound: 57.4904413
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -10.0193787, 42.8943863, -8.3100739, 35.6519890, -45.6713676, 51.2044563
1: -12.8247375, 48.5298729, -10.6276932, 40.3892288, -53.2139626, 59.1575661
2: -12.5208769, 48.2823448, -10.4366865, 39.9314880, -52.4523582, 58.7190323
3: -21.9548664, 51.6154327, -18.2305756, 43.0719185, -65.0267792, 69.8460083
4: -20.2769165, 49.6436996, -16.8955002, 41.1396370, -61.4165535, 66.5391998

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2967605, upper bound: 57.4165580
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2967605, upper bound: 57.4165580
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -10.0372343, 42.1927071, -9.1575928, 38.8744278, -48.9116554, 51.3502960
1: -12.7809610, 47.7437553, -11.6959438, 44.0212021, -56.8021622, 59.4396973
2: -12.5463152, 47.4627686, -11.4683819, 43.6604576, -56.2067719, 58.9311523
3: -21.6898041, 50.8244820, -19.9659348, 46.8748016, -68.5645981, 70.7904053
4: -20.1472874, 48.8353882, -18.4506874, 44.9865723, -65.1338577, 67.2860718

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5135716, upper bound: 57.5083693
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5135716, upper bound: 57.5112195
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -10.0193787, 42.8943863, -7.9867325, 34.2763634, -44.2957420, 50.8811073
1: -12.8247375, 48.5298729, -10.1965561, 38.8652573, -51.6899948, 58.7264252
2: -12.5208769, 48.2823448, -10.0668869, 38.3097916, -50.8306618, 58.3492317
3: -21.9548664, 51.6154327, -17.4640160, 41.5618553, -63.5167236, 69.0794525
4: -20.2769165, 49.6436996, -16.3867416, 39.4446716, -59.7215881, 66.0304413

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3021520, upper bound: 57.4303925
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3021520, upper bound: 57.4338251
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -10.0372343, 42.1927071, -8.8118200, 37.3577957, -47.3950272, 51.0045280
1: -12.7809610, 47.7437553, -11.2442751, 42.3229713, -55.1039314, 58.9880295
2: -12.5463152, 47.4627686, -11.0679474, 41.8824844, -54.4287987, 58.5307159
3: -21.6898041, 50.8244820, -19.1606712, 45.1869125, -66.8767090, 69.9851532
4: -20.1472874, 48.8353882, -17.8871288, 43.1344337, -63.2817154, 66.7224960

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5395300, upper bound: 57.5311056
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5395300, upper bound: 57.5523846
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -10.6957531, 44.9465866, -11.0297852, 46.1416092, -56.8373566, 55.9763718
1: -13.6549959, 50.8562546, -14.0756989, 52.2037125, -65.8587036, 64.9319534
2: -13.3063459, 50.7025528, -13.7119274, 52.0922241, -65.3985672, 64.4144440
3: -23.1477661, 54.0001144, -23.8188839, 55.4197121, -78.5674744, 77.8190002
4: -21.2595654, 52.1365585, -21.8664837, 53.5777702, -74.8373337, 74.0030441

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -57.2165477, upper bound: 57.0956875
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2165477, upper bound: 57.4665788
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -11.0847101, 47.3818054, -9.6080275, 40.5105820, -51.5952911, 56.9898262
1: -14.2560978, 53.6242714, -12.2365494, 45.8477631, -60.1038589, 65.8608093
2: -13.7948971, 53.5095329, -12.0228996, 45.5226021, -59.3174934, 65.5324249
3: -24.3539143, 56.8776131, -20.8062382, 48.8360825, -73.1899796, 77.6838531
4: -22.2534161, 54.9756432, -19.3448238, 46.8458099, -69.0992203, 74.3204498

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -57.0553677, upper bound: 57.1244992
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.0553677, upper bound: 57.3156531
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -10.6957531, 44.9465866, -10.4046993, 43.4960098, -54.1917572, 55.3512764
1: -13.6549959, 50.8562546, -13.2431650, 49.2144547, -62.8694344, 64.0994186
2: -13.3063459, 50.7025528, -12.9902172, 48.9800606, -62.2864075, 63.6927376
3: -23.1477661, 54.0001144, -22.4255047, 52.3740425, -75.5218048, 76.4256210
4: -21.2595654, 52.1365585, -20.8130035, 50.4081459, -71.6677094, 72.9495544

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -57.2091961, upper bound: 57.1524313
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2091961, upper bound: 57.4904413
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -10.0372343, 42.1927071, -11.0297852, 46.1416092, -56.1788368, 53.2224922
1: -12.7809610, 47.7437553, -14.0756989, 52.2037125, -64.9846725, 61.8194542
2: -12.5463152, 47.4627686, -13.7119274, 52.0922241, -64.6385422, 61.1746864
3: -21.6898041, 50.8244820, -23.8188839, 55.4197121, -77.1095123, 74.6433640
4: -20.1472874, 48.8353882, -21.8664837, 53.5777702, -73.7250519, 70.7018738

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3156531, upper bound: 57.1381061
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3156531, upper bound: 57.4817208
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -10.0193787, 42.8943863, -9.6080275, 40.5105820, -50.5299606, 52.5024147
1: -12.8247375, 48.5298729, -12.2365494, 45.8477631, -58.6725006, 60.7664185
2: -12.5208769, 48.2823448, -12.0228996, 45.5226021, -58.0434799, 60.3052368
3: -21.9548664, 51.6154327, -20.8062382, 48.8360825, -70.7909317, 72.4216690
4: -20.2769165, 49.6436996, -19.3448238, 46.8458099, -67.1227112, 68.9885178

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
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
time: 1.02 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2045298, upper bound: 57.3737711
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -10.0372343, 42.1927071, -10.4046993, 43.4960098, -53.5332336, 52.5974007
1: -12.7809610, 47.7437553, -13.2431650, 49.2144547, -61.9954071, 60.9869194
2: -12.5463152, 47.4627686, -12.9902172, 48.9800606, -61.5263748, 60.4529800
3: -21.6898041, 50.8244820, -22.4255047, 52.3740425, -74.0638428, 73.2499847
4: -20.1472874, 48.8353882, -20.8130035, 50.4081459, -70.5554123, 69.6483841

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4422203, upper bound: 57.3053738
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4422203, upper bound: 57.5620557
time: 0.67 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.70 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.4801864, upper bound: 57.4402933
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.5011117, upper bound: 57.4986063
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.5087887, upper bound: 57.5087887
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.5087887, upper bound: 57.5087887
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.4921367, upper bound: 57.4743048
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.5076615, upper bound: 57.5197585
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.5141803, upper bound: 57.5226232
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.5141803, upper bound: 57.5315670
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.4874880, upper bound: 57.4417602
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.5084133, upper bound: 57.5000732
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.5313612, upper bound: 57.5166199
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.5313612, upper bound: 57.5166199
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.4962781, upper bound: 57.4639298
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.5136952, upper bound: 57.5136952
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.5411622, upper bound: 57.5318641
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.5411622, upper bound: 57.5497019
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.3036922, upper bound: 57.1357383
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.3036922, upper bound: 57.4404028
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.3036922, upper bound: 57.1357383
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.3036922, upper bound: 57.4753854
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.4815300, upper bound: 57.4574505
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.4947398, upper bound: 57.5011140
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.4165580, upper bound: 57.2967605
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.4165580, upper bound: 57.5284482
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.3201454, upper bound: 57.1423158
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.3201454, upper bound: 57.3370890
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.3271205, upper bound: 57.1443533
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.3271205, upper bound: 57.4781048
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.4670644, upper bound: 57.3944944
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.4081399, upper bound: 57.3521439
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.4437227, upper bound: 57.3060635
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.4437227, upper bound: 57.5497017
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.1357383, upper bound: 57.3036922
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.1357383, upper bound: 57.3036922
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.4404028, upper bound: 57.4563171
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.4404028, upper bound: 57.4822205
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.1423158, upper bound: 57.3201454
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.1423158, upper bound: 57.3271205
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.3370890, upper bound: 57.4334922
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.3370890, upper bound: 57.4904413
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.2967605, upper bound: 57.4165580
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.2967605, upper bound: 57.4165580
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.5135716, upper bound: 57.5083693
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.5135716, upper bound: 57.5112195
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.3021520, upper bound: 57.4303925
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.3021520, upper bound: 57.4338251
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.5395300, upper bound: 57.5311056
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.5395300, upper bound: 57.5523846
IS_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.2165477, upper bound: 57.0956875
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.2165477, upper bound: 57.4665788
IS_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.0553677, upper bound: 57.1244992
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.0553677, upper bound: 57.3156531
IS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.2091961, upper bound: 57.1524313
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.2091961, upper bound: 57.4904413
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.3156531, upper bound: 57.1381061
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.3156531, upper bound: 57.4817208
IS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.2045298, upper bound: 57.2045298
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.2045298, upper bound: 57.3737711
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.4422203, upper bound: 57.3053738
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.70
Output dim: 0, lower bound: -57.4422203, upper bound: 57.5620557

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -8.7669353, 38.1990242, -5.4886818, 25.9273338, -34.6942635, 43.6877060
1: -11.2745037, 43.2634735, -6.9812737, 29.5331364, -40.8076286, 50.2447472
2: -11.0006275, 42.8983345, -7.0669203, 28.6377640, -39.6383896, 49.9652557
3: -19.5009098, 46.0489769, -12.3952608, 31.6303558, -51.1312637, 58.4442368
4: -17.9422951, 44.1722069, -11.9016552, 29.3206253, -47.2629128, 56.0738602

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4235198, upper bound: 57.4200260
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4235198, upper bound: 57.4402933
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -7.7099409, 34.2308578, -6.6432877, 29.6070499, -37.3169899, 40.8741455
1: -9.9192944, 38.7935486, -8.4892225, 33.6587601, -43.5780525, 47.2827721
2: -9.7258482, 38.3159447, -8.4573994, 32.9010658, -42.6269073, 46.7733459
3: -17.2796021, 41.3616943, -14.6922836, 36.1012115, -53.3808136, 56.0539780
4: -15.9834738, 39.4624023, -13.8673687, 33.8380470, -49.8215141, 53.3297691

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4444451, upper bound: 57.4783391
time: 0.52 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4444451, upper bound: 57.4986064
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.8402529, 37.7630539, -8.7669353, 38.1990242, -47.0392761, 46.5299797
1: -11.2974596, 42.7681274, -11.2745037, 43.2634735, -54.5609322, 54.0426254
2: -11.0862007, 42.3602257, -11.0006275, 42.8983345, -53.9845352, 53.3608551
3: -19.3324356, 45.5596771, -19.5009098, 46.0489769, -65.3814087, 65.0605850
4: -17.8893318, 43.6359253, -17.9422951, 44.1722069, -62.0615234, 61.5782166

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4364782, upper bound: 57.4738660
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4947912, upper bound: 57.4947913
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.8402529, 37.7630539, -8.8402529, 37.7630539, -46.6033058, 46.6033058
1: -11.2974596, 42.7681274, -11.2974596, 42.7681274, -54.0655861, 54.0655861
2: -11.0862007, 42.3602257, -11.0862007, 42.3602257, -53.4464264, 53.4464264
3: -19.3324356, 45.5596771, -19.3324356, 45.5596771, -64.8921127, 64.8921127
4: -17.8893318, 43.6359253, -17.8893318, 43.6359253, -61.5252380, 61.5252419

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4364782, upper bound: 57.4738660
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4947912, upper bound: 57.4947913
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -8.7669353, 38.1990242, -5.1475730, 24.5281429, -33.2950745, 43.3465958
1: -11.2745037, 43.2634735, -6.5067697, 28.0255165, -39.3000107, 49.7702446
2: -11.0006275, 42.8983345, -6.6766939, 27.0021133, -38.0027390, 49.5750275
3: -19.5009098, 46.0489769, -11.5436974, 30.0893459, -49.5902557, 57.5926743
4: -17.9422951, 44.1722069, -11.3664494, 27.5474873, -45.4897804, 55.5386505

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4350126, upper bound: 57.4527563
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4350126, upper bound: 57.4743048
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -7.7099409, 34.2308578, -6.6124506, 29.4659061, -37.1758461, 40.8433075
1: -9.9192944, 38.7935486, -8.4423676, 33.5392151, -43.4585037, 47.2359161
2: -9.7258482, 38.3159447, -8.4463482, 32.6769257, -42.4027710, 46.7622910
3: -17.2796021, 41.3616943, -14.5717545, 36.0493965, -53.3289948, 55.9334450
4: -15.9834738, 39.4624023, -13.9092426, 33.5896416, -49.5731125, 53.3716431

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4444451, upper bound: 57.4989698
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4507852, upper bound: 57.5197585
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.8402529, 37.7630539, -8.0646629, 35.2838440, -44.1240959, 45.8277092
1: -11.2974596, 42.7681274, -10.3480911, 39.9815979, -51.2790565, 53.1162186
2: -11.0862007, 42.3602257, -10.1572323, 39.4773331, -50.5635338, 52.5174561
3: -19.3324356, 45.5596771, -17.9082813, 42.6824608, -62.0148964, 63.4679565
4: -17.8893318, 43.6359253, -16.6487427, 40.6530380, -58.5423698, 60.2846603

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4417602, upper bound: 57.4874880
time: 0.52 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5000732, upper bound: 57.5084133
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.8402529, 37.7630539, -8.4693804, 36.1834831, -45.0237350, 46.2324257
1: -11.2974596, 42.7681274, -10.8127460, 41.0070839, -52.3045425, 53.5808716
2: -11.0862007, 42.3602257, -10.6570034, 40.5084305, -51.5946312, 53.0172272
3: -19.3324356, 45.5596771, -18.4758568, 43.8035660, -63.1359978, 64.0355377
4: -17.8893318, 43.6359253, -17.2884102, 41.7039909, -59.5933228, 60.9243355

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4417602, upper bound: 57.4903789
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5000732, upper bound: 57.5116376
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -8.0646629, 35.2838440, -5.4886818, 25.9273338, -33.9919891, 40.7725258
1: -10.3480911, 39.9815979, -6.9812737, 29.5331364, -39.8812256, 46.9628716
2: -10.1572323, 39.4773331, -7.0669203, 28.6377640, -38.7949982, 46.5442543
3: -17.9082813, 42.6824608, -12.3952608, 31.6303558, -49.5386353, 55.0777206
4: -16.6487427, 40.6530380, -11.9016552, 29.3206253, -45.9693604, 52.5546951

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4395102, upper bound: 57.4250266
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4395102, upper bound: 57.4417602
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -7.4394846, 32.9472542, -6.6432877, 29.6070499, -37.0465355, 39.5905418
1: -9.5418024, 37.3610992, -8.4892225, 33.6587601, -43.2005615, 45.8503227
2: -9.4033060, 36.7727203, -8.4573994, 32.9010658, -42.3043671, 45.2301178
3: -16.5861855, 39.9446564, -14.6922836, 36.1012115, -52.6873970, 54.6369400
4: -15.5030107, 37.8687439, -13.8673687, 33.8380470, -49.3410492, 51.7361069

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4604355, upper bound: 57.4833397
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4604355, upper bound: 57.5000732
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.4693804, 36.1834831, -8.7669353, 38.1990242, -46.6684036, 44.9504089
1: -10.8127460, 41.0070839, -11.2745037, 43.2634735, -54.0762177, 52.2815781
2: -10.6570034, 40.5084305, -11.0006275, 42.8983345, -53.5553360, 51.5090561
3: -18.4758568, 43.8035660, -19.5009098, 46.0489769, -64.5248337, 63.3044739
4: -17.2884102, 41.7039909, -17.9422951, 44.1722069, -61.4606171, 59.6462860

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4743048, upper bound: 57.4880902
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5183954, upper bound: 57.5024117
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.4693804, 36.1834831, -8.8402529, 37.7630539, -46.2324257, 45.0237350
1: -10.8127460, 41.0070839, -11.2974596, 42.7681274, -53.5808716, 52.3045425
2: -10.6570034, 40.5084305, -11.0862007, 42.3602257, -53.0172272, 51.5946312
3: -18.4758568, 43.8035660, -19.3324356, 45.5596771, -64.0355377, 63.1359978
4: -17.2884102, 41.7039909, -17.8893318, 43.6359253, -60.9243240, 59.5933228

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4743048, upper bound: 57.4880902
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5183954, upper bound: 57.5024117
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -8.0646629, 35.2838440, -5.1475730, 24.5281429, -32.5928040, 40.4314156
1: -10.3480911, 39.9815979, -6.5067697, 28.0255165, -38.3736076, 46.4883690
2: -10.1572323, 39.4773331, -6.6766939, 27.0021133, -37.1593475, 46.1540260
3: -17.9082813, 42.6824608, -11.5436974, 30.0893459, -47.9976273, 54.2261581
4: -16.6487427, 40.6530380, -11.3664494, 27.5474873, -44.1962242, 52.0194817

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4489839, upper bound: 57.4489839
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4489839, upper bound: 57.4639298
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -7.4394846, 32.9472542, -6.6124506, 29.4659061, -36.9053879, 39.5597000
1: -9.5418024, 37.3610992, -8.4423676, 33.5392151, -43.0810165, 45.8034668
2: -9.4033060, 36.7727203, -8.4463482, 32.6769257, -42.0802307, 45.2190704
3: -16.5861855, 39.9446564, -14.5717545, 36.0493965, -52.6355820, 54.5164108
4: -15.5030107, 37.8687439, -13.9092426, 33.5896416, -49.0926476, 51.7779846

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4659216, upper bound: 57.4976213
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4659216, upper bound: 57.5136952
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.4693804, 36.1834831, -8.0646629, 35.2838440, -43.7532234, 44.2481422
1: -10.8127460, 41.0070839, -10.3480911, 39.9815979, -50.7943420, 51.3551750
2: -10.6570034, 40.5084305, -10.1572323, 39.4773331, -50.1343384, 50.6656647
3: -18.4758568, 43.8035660, -17.9082813, 42.6824608, -61.1583176, 61.7118454
4: -17.2884102, 41.7039909, -16.6487427, 40.6530380, -57.9414482, 58.3527298

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4869772, upper bound: 57.5043538
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5259809, upper bound: 57.5172517
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.4693804, 36.1834831, -8.4693804, 36.1834831, -44.6528587, 44.6528587
1: -10.8127460, 41.0070839, -10.8127460, 41.0070839, -51.8198318, 51.8198318
2: -10.6570034, 40.5084305, -10.6570034, 40.5084305, -51.1654358, 51.1654320
3: -18.4758568, 43.8035660, -18.4758568, 43.8035660, -62.2794228, 62.2794228
4: -17.2884102, 41.7039909, -17.2884102, 41.7039909, -58.9924011, 58.9924011

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4869772, upper bound: 57.5168833
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5259809, upper bound: 57.5316336
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -8.7669353, 38.1990242, -11.0243721, 47.1488724, -55.9158020, 49.2233963
1: -11.2745037, 43.2634735, -14.1803865, 53.3598557, -64.6343460, 57.4438591
2: -11.0006275, 42.8983345, -13.7203350, 53.2445908, -64.2452087, 56.6186676
3: -19.5009098, 46.0489769, -24.2248249, 56.5950623, -76.0959549, 70.2737961
4: -17.9422951, 44.1722069, -22.1343040, 54.7041283, -72.6464157, 66.3065033

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3107140, upper bound: 57.1401241
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -57.2725323, upper bound: 57.1238339
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -8.7669353, 38.1990242, -10.6957531, 44.9465866, -53.7135086, 48.8947754
1: -11.2745037, 43.2634735, -13.6549959, 50.8562546, -62.1307487, 56.9184685
2: -11.0006275, 42.8983345, -13.3063459, 50.7025528, -61.7031593, 56.2046814
3: -19.5009098, 46.0489769, -23.1477661, 54.0001144, -73.5010223, 69.1967468
4: -17.9422951, 44.1722069, -21.2595654, 52.1365585, -70.0788574, 65.4317703

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3107140, upper bound: 57.4358203
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2725323, upper bound: 57.4023212
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.8402529, 37.7630539, -11.0847101, 47.3818054, -56.2220535, 48.8477554
1: -11.2974596, 42.7681274, -14.2560978, 53.6242714, -64.9217224, 57.0242233
2: -11.0862007, 42.3602257, -13.7948971, 53.5095329, -64.5957336, 56.1551208
3: -19.3324356, 45.5596771, -24.3539143, 56.8776131, -76.2100525, 69.9135895
4: -17.8893318, 43.6359253, -22.2534161, 54.9756432, -72.8649521, 65.8893433

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -57.2759431, upper bound: 57.1095837
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3012249, upper bound: 57.1329330
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.8402529, 37.7630539, -10.6957531, 44.9465866, -53.7868385, 48.4588013
1: -11.2974596, 42.7681274, -13.6549959, 50.8562546, -62.1537056, 56.4231224
2: -11.0862007, 42.3602257, -13.3063459, 50.7025528, -61.7887383, 55.6665726
3: -19.3324356, 45.5596771, -23.1477661, 54.0001144, -73.3325500, 68.7074432
4: -17.8893318, 43.6359253, -21.2595654, 52.1365585, -70.0258789, 64.8954926

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2759431, upper bound: 57.4581064
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3012249, upper bound: 57.4641336
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -8.7669353, 38.1990242, -6.6672854, 30.3811951, -39.1481247, 44.8663101
1: -11.2745037, 43.2634735, -8.4935579, 34.4831467, -45.7576447, 51.7570305
2: -11.0006275, 42.8983345, -8.5128460, 33.7376404, -44.7382660, 51.4111786
3: -19.5009098, 46.0489769, -14.8610125, 36.9102592, -56.4111710, 60.9099808
4: -17.9422951, 44.1722069, -14.1184769, 34.5963783, -52.5386734, 58.2906799

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4226292, upper bound: 57.4346183
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4226292, upper bound: 57.4574505
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -7.7099409, 34.2308578, -7.8887362, 34.1600571, -41.8699989, 42.1195946
1: -9.9192944, 38.7935486, -10.0579615, 38.7469788, -48.6662674, 48.8515091
2: -9.7258482, 38.3159447, -9.9619560, 38.1311836, -47.8570290, 48.2778969
3: -17.2796021, 41.3616943, -17.1808949, 41.4463387, -58.7259407, 58.5425873
4: -15.9834738, 39.4624023, -16.1190166, 39.1840782, -55.1675529, 55.5814209

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4352799, upper bound: 57.4776459
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4352799, upper bound: 57.5011141
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.8402529, 37.7630539, -10.0193787, 42.8943863, -51.7346382, 47.7824326
1: -11.2974596, 42.7681274, -12.8247375, 48.5298729, -59.8273315, 55.5928650
2: -11.0862007, 42.3602257, -12.5208769, 48.2823448, -59.3685455, 54.8810959
3: -19.3324356, 45.5596771, -21.9548664, 51.6154327, -70.9478683, 67.5145416
4: -17.8893318, 43.6359253, -20.2769165, 49.6436996, -67.5330276, 63.9128418

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3992415, upper bound: 57.2795275
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3992415, upper bound: 57.2945642
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.8402529, 37.7630539, -10.0372343, 42.1927071, -51.0329590, 47.8002815
1: -11.2974596, 42.7681274, -12.7809610, 47.7437553, -59.0412064, 55.5490875
2: -11.0862007, 42.3602257, -12.5463152, 47.4627686, -58.5489693, 54.9065399
3: -19.3324356, 45.5596771, -21.6898041, 50.8244820, -70.1569214, 67.2494736
4: -17.8893318, 43.6359253, -20.1472874, 48.8353882, -66.7247009, 63.7832031

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3992415, upper bound: 57.4948922
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3992415, upper bound: 57.5044127
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -8.0646629, 35.2838440, -11.0243721, 47.1488724, -55.2135353, 46.3082123
1: -10.3480911, 39.9815979, -14.1803865, 53.3598557, -63.7079315, 54.1619835
2: -10.1572323, 39.4773331, -13.7203350, 53.2445908, -63.4018021, 53.1976700
3: -17.9082813, 42.6824608, -24.2248249, 56.5950623, -74.5033188, 66.9072876
4: -16.6487427, 40.6530380, -22.1343040, 54.7041283, -71.3528748, 62.7873344

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2963110, upper bound: 57.1176528
time: 0.55 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3201454, upper bound: 57.1423158
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -8.0646629, 35.2838440, -10.6957531, 44.9465866, -53.0112457, 45.9795952
1: -10.3480911, 39.9815979, -13.6549959, 50.8562546, -61.2043457, 53.6365891
2: -10.1572323, 39.4773331, -13.3063459, 50.7025528, -60.8597527, 52.7836800
3: -17.9082813, 42.6824608, -23.1477661, 54.0001144, -71.9083939, 65.8302307
4: -16.6487427, 40.6530380, -21.2595654, 52.1365585, -68.7852936, 61.9126053

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2963110, upper bound: 57.2010017
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3201454, upper bound: 57.3370890
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.4693804, 36.1834831, -11.0847101, 47.3818054, -55.8511772, 47.2681885
1: -10.8127460, 41.0070839, -14.2560978, 53.6242714, -64.4370117, 55.2631836
2: -10.6570034, 40.5084305, -13.7948971, 53.5095329, -64.1665344, 54.3033257
3: -18.4758568, 43.8035660, -24.3539143, 56.8776131, -75.3534698, 68.1574783
4: -17.2884102, 41.7039909, -22.2534161, 54.9756432, -72.2640533, 63.9574051

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2989039, upper bound: 57.1144925
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3244520, upper bound: 57.1415367
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.4693804, 36.1834831, -10.6957531, 44.9465866, -53.4159584, 46.8792305
1: -10.8127460, 41.0070839, -13.6549959, 50.8562546, -61.6689987, 54.6620789
2: -10.6570034, 40.5084305, -13.3063459, 50.7025528, -61.3595390, 53.8147736
3: -18.4758568, 43.8035660, -23.1477661, 54.0001144, -72.4759674, 66.9513321
4: -17.2884102, 41.7039909, -21.2595654, 52.1365585, -69.4249725, 62.9635544

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2989039, upper bound: 57.4590370
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3244520, upper bound: 57.4641336
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -8.0646629, 35.2838440, -6.6672854, 30.3811951, -38.4458542, 41.9511299
1: -10.3480911, 39.9815979, -8.4935579, 34.4831467, -44.8312378, 48.4751549
2: -10.1572323, 39.4773331, -8.5128460, 33.7376404, -43.8948746, 47.9901810
3: -17.9082813, 42.6824608, -14.8610125, 36.9102592, -54.8185425, 57.5434685
4: -16.6487427, 40.6530380, -14.1184769, 34.5963783, -51.2451210, 54.7715149

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3699886, upper bound: 57.3505377
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3699886, upper bound: 57.3521451
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -7.4394846, 32.9472542, -7.8887362, 34.1600571, -41.5995407, 40.8359871
1: -9.5418024, 37.3610992, -10.0579615, 38.7469788, -48.2887802, 47.4190598
2: -9.4033060, 36.7727203, -9.9619560, 38.1311836, -47.5344887, 46.7346764
3: -16.5861855, 39.9446564, -17.1808949, 41.4463387, -58.0325203, 57.1255493
4: -15.5030107, 37.8687439, -16.1190166, 39.1840782, -54.6870880, 53.9877586

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3699886, upper bound: 57.3505377
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3699886, upper bound: 57.3521450
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.4693804, 36.1834831, -10.0193787, 42.8943863, -51.3637657, 46.2028618
1: -10.8127460, 41.0070839, -12.8247375, 48.5298729, -59.3426208, 53.8318214
2: -10.6570034, 40.5084305, -12.5208769, 48.2823448, -58.9393463, 53.0293007
3: -18.4758568, 43.8035660, -21.9548664, 51.6154327, -70.0912933, 65.7584305
4: -17.2884102, 41.7039909, -20.2769165, 49.6436996, -66.9321136, 61.9809074

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243064, upper bound: 57.2850175
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4415822, upper bound: 57.3038770
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.4693804, 36.1834831, -10.0372343, 42.1927071, -50.6620827, 46.2207108
1: -10.8127460, 41.0070839, -12.7809610, 47.7437553, -58.5564957, 53.7880440
2: -10.6570034, 40.5084305, -12.5463152, 47.4627686, -58.1197739, 53.0547447
3: -18.4758568, 43.8035660, -21.6898041, 50.8244820, -69.3003387, 65.4933624
4: -17.2884102, 41.7039909, -20.1472874, 48.8353882, -66.1237946, 61.8512802

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4243064, upper bound: 57.5386526
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4415822, upper bound: 57.5483706
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -11.0847101, 47.3818054, -8.7669353, 38.1990242, -49.2837334, 56.1487274
1: -14.2560978, 53.6242714, -11.2745037, 43.2634735, -57.5195694, 64.8987656
2: -13.7948971, 53.5095329, -11.0006275, 42.8983345, -56.6932297, 64.5101624
3: -24.3539143, 56.8776131, -19.5009098, 46.0489769, -70.4028931, 76.3785248
4: -22.2534161, 54.9756432, -17.9422951, 44.1722069, -66.4256210, 72.9179382

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -56.2836160, upper bound: 56.4054849
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.1337538, upper bound: 57.3029626
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -11.0847101, 47.3818054, -8.8402529, 37.7630539, -48.8477554, 56.2220535
1: -14.2560978, 53.6242714, -11.2974596, 42.7681274, -57.0242233, 64.9217224
2: -13.7948971, 53.5095329, -11.0862007, 42.3602257, -56.1551208, 64.5957260
3: -24.3539143, 56.8776131, -19.3324356, 45.5596771, -69.9135895, 76.2100525
4: -22.2534161, 54.9756432, -17.8893318, 43.6359253, -65.8893433, 72.8649521

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -56.2836160, upper bound: 56.4054849
time: 1.01 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.1337538, upper bound: 57.3029626
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -10.6957531, 44.9465866, -8.7669353, 38.1990242, -48.8947754, 53.7135124
1: -13.6549959, 50.8562546, -11.2745037, 43.2634735, -56.9184685, 62.1307526
2: -13.3063459, 50.7025528, -11.0006275, 42.8983345, -56.2046814, 61.7031593
3: -23.1477661, 54.0001144, -19.5009098, 46.0489769, -69.1967468, 73.5010223
4: -21.2595654, 52.1365585, -17.9422951, 44.1722069, -65.4317703, 70.0788498

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -56.5949070, upper bound: 56.7768506
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4403598, upper bound: 57.4561414
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -10.6957531, 44.9465866, -8.8402529, 37.7630539, -48.4588013, 53.7868347
1: -13.6549959, 50.8562546, -11.2974596, 42.7681274, -56.4231224, 62.1537132
2: -13.3063459, 50.7025528, -11.0862007, 42.3602257, -55.6665726, 61.7887306
3: -23.1477661, 54.0001144, -19.3324356, 45.5596771, -68.7074432, 73.3325500
4: -21.2595654, 52.1365585, -17.8893318, 43.6359253, -64.8954926, 70.0258713

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -56.5949070, upper bound: 56.9842616
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4403598, upper bound: 57.4820637
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -11.0847101, 47.3818054, -8.0646629, 35.2838440, -46.3685532, 55.4464607
1: -14.2560978, 53.6242714, -10.3480911, 39.9815979, -54.2376938, 63.9723549
2: -13.7948971, 53.5095329, -10.1572323, 39.4773331, -53.2722282, 63.6667595
3: -24.3539143, 56.8776131, -17.9082813, 42.6824608, -67.0363770, 74.7858963
4: -22.2534161, 54.9756432, -16.6487427, 40.6530380, -62.9064560, 71.6243820

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -57.0987839, upper bound: 57.2797060
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 46
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 20

Time for candidate selection: 6.61 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 39

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -57.0741102, upper bound: 57.2398042
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -57.0878741, upper bound: 57.2326454
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -57.1021395, upper bound: 57.2403976
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -11.0847101, 47.3818054, -8.4693804, 36.1834831, -47.2681885, 55.8511772
1: -14.2560978, 53.6242714, -10.8127460, 41.0070839, -55.2631798, 64.4370193
2: -13.7948971, 53.5095329, -10.6570034, 40.5084305, -54.3033218, 64.1665344
3: -24.3539143, 56.8776131, -18.4758568, 43.8035660, -68.1574783, 75.3534698
4: -22.2534161, 54.9756432, -17.2884102, 41.7039909, -63.9574051, 72.2640533

Time for backsubstitution: 2.37 seconds
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=66.57380676269531
rel_dist={0: [-57.5687467976788, 57.5687467976788]}

## Binary search (step 2) starts
Candidate diff: 0.0625000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

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
time: 0.56 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.27 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.27
Output dim: 0, lower bound: -57.5526292, upper bound: 57.5552693
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.27
Output dim: 0, lower bound: -57.5620108, upper bound: 57.5620108

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -10.1770201, 42.4380188, -11.4069080, 47.1075592, -57.2845764, 53.8449249
1: -12.9492474, 48.0443954, -14.4834604, 53.3023415, -66.2515869, 62.5278549
2: -12.7204800, 47.7395439, -14.2116318, 53.1170883, -65.8375702, 61.9511719
3: -21.9290562, 51.2056084, -24.4205017, 56.6857185, -78.6147614, 75.6261063
4: -20.4464874, 49.1660233, -22.7181416, 54.7416573, -75.1881332, 71.8841629

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4790544, upper bound: 57.4728293
time: 0.55 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5526292, upper bound: 57.5552693
time: 0.60 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -11.9793425, 49.4708633, -12.7021961, 52.2306023, -64.2099457, 62.1730576
1: -15.2106190, 55.9559326, -16.1179504, 59.0786743, -74.2892914, 72.0738754
2: -14.9106045, 55.8369446, -15.7900686, 59.0102730, -73.9208755, 71.6270142
3: -25.6317139, 59.4555740, -27.1216049, 62.7400017, -88.3717117, 86.5771790
4: -23.7928352, 57.5261345, -25.1618710, 60.8016586, -84.5944977, 82.6879959

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5552693, upper bound: 57.5526292
time: 0.52 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5552693, upper bound: 57.5620108
time: 0.62 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.29 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.29
Output dim: 0, lower bound: -57.4790544, upper bound: 57.4728293
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.29
Output dim: 0, lower bound: -57.5526292, upper bound: 57.5552693
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.29
Output dim: 0, lower bound: -57.5552693, upper bound: 57.5526292
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.29
Output dim: 0, lower bound: -57.5552693, upper bound: 57.5620108

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -9.0654306, 38.2294083, -10.4684792, 43.8125954, -52.8780251, 48.6978836
1: -11.5589437, 43.2975159, -13.3555183, 49.5893250, -61.1482697, 56.6530342
2: -11.3617239, 42.8905373, -13.0243645, 49.3946419, -60.7563667, 55.9148941
3: -19.6654778, 46.1984787, -22.6225986, 52.6877899, -72.3532715, 68.8210602
4: -18.3505554, 44.1525192, -20.8169117, 50.8262405, -69.1767883, 64.9694290

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2985682, upper bound: 57.2063044
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4787478, upper bound: 57.4724487
time: 0.58 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -9.6058359, 40.3157463, -9.8341084, 41.1668472, -50.7726746, 50.1498489
1: -12.2354774, 45.6513290, -12.5149050, 46.5993690, -58.8348465, 58.1662292
2: -12.0278358, 45.2923660, -12.2963152, 46.2908249, -58.3186569, 57.5886803
3: -20.7736835, 48.6870728, -21.2173576, 49.6434097, -70.4170914, 69.9044113
4: -19.3757629, 46.6283188, -19.7471123, 47.6589088, -67.0346680, 66.3754272

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4934149, upper bound: 57.5019506
time: 0.54 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4934149, upper bound: 57.5552693
time: 0.59 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -11.9793425, 49.4708633, -10.1770201, 42.4380188, -54.4173622, 59.6478844
1: -15.2106190, 55.9559326, -12.9492474, 48.0443954, -63.2550125, 68.9051743
2: -14.9106045, 55.8369446, -12.7204800, 47.7395439, -62.6501465, 68.5574265
3: -25.6317139, 59.4555740, -21.9290562, 51.2056084, -76.8373108, 81.3846283
4: -23.7928352, 57.5261345, -20.4464874, 49.1660233, -72.9588547, 77.9725952

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4728293, upper bound: 57.4790544
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5552692, upper bound: 57.5526292
time: 0.61 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -11.9793425, 49.4708633, -11.9793425, 49.4708633, -61.4502068, 61.4502068
1: -15.2106190, 55.9559326, -15.2106190, 55.9559326, -71.1665497, 71.1665497
2: -14.9106045, 55.8369446, -14.9106045, 55.8369446, -70.7475510, 70.7475510
3: -25.6317139, 59.4555740, -25.6317139, 59.4555740, -85.0872879, 85.0872879
4: -23.7928352, 57.5261345, -23.7928352, 57.5261345, -81.3189545, 81.3189621

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4728293, upper bound: 57.4790544
time: 0.53 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5552693, upper bound: 57.5619695
time: 0.58 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.34 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.34
Output dim: 0, lower bound: -57.2985682, upper bound: 57.2063044
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.34
Output dim: 0, lower bound: -57.4787478, upper bound: 57.4724487
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.34
Output dim: 0, lower bound: -57.4934149, upper bound: 57.5019506
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.34
Output dim: 0, lower bound: -57.4934149, upper bound: 57.5552693
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.34
Output dim: 0, lower bound: -57.4728293, upper bound: 57.4790544
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.34
Output dim: 0, lower bound: -57.5552692, upper bound: 57.5526292
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.34
Output dim: 0, lower bound: -57.4728293, upper bound: 57.4790544
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.34
Output dim: 0, lower bound: -57.5552693, upper bound: 57.5619695

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -8.1608181, 35.4994659, -8.9344349, 38.1125145, -46.2733307, 44.4338989
1: -10.4707947, 40.2127953, -11.4183254, 43.1567764, -53.6275635, 51.6311188
2: -10.2669220, 39.7491150, -11.1656008, 42.7920761, -53.0589981, 50.9147148
3: -18.0829659, 42.9250832, -19.5120735, 45.9320831, -64.0150452, 62.4371567
4: -16.8136044, 40.9275131, -18.0235748, 44.0175133, -60.8311119, 58.9510880

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2985682, upper bound: 57.2063044
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2985682, upper bound: 57.2063044
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8.7259798, 37.0416527, -10.4643745, 43.7978134, -52.5237923, 47.5060234
1: -11.1310406, 41.9613190, -13.3503437, 49.5726395, -60.7036819, 55.3116608
2: -10.9537401, 41.5014725, -13.0193863, 49.3774872, -60.3312263, 54.5208549
3: -18.9851074, 44.7977524, -22.6143150, 52.6702499, -71.6553574, 67.4120636
4: -17.7529068, 42.7079048, -20.8094139, 50.8084946, -68.5614014, 63.5173187

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4787478, upper bound: 57.4724487
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4787478, upper bound: 57.4724487
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -9.1575928, 38.8744278, -9.8341084, 41.1668472, -50.3244362, 48.7085342
1: -11.6959438, 44.0212021, -12.5149050, 46.5993690, -58.2953110, 56.5361061
2: -11.4683819, 43.6604576, -12.2963152, 46.2908249, -57.7592087, 55.9567680
3: -19.9659348, 46.8748016, -21.2173576, 49.6434097, -69.6093292, 68.0921555
4: -18.4506874, 44.9865723, -19.7471123, 47.6589088, -66.1095963, 64.7336807

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4751695, upper bound: 57.5019506
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4751695, upper bound: 57.5019506
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.8118200, 37.3577957, -9.8341084, 41.1668472, -49.9786682, 47.1919022
1: -11.2442751, 42.3229713, -12.5149050, 46.5993690, -57.8436432, 54.8378754
2: -11.0679474, 41.8824844, -12.2963152, 46.2908249, -57.3587685, 54.1787987
3: -19.1606712, 45.1869125, -21.2173576, 49.6434097, -68.8040771, 66.4042587
4: -17.8871288, 43.1344337, -19.7471123, 47.6589088, -65.5460358, 62.8815346

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4751695, upper bound: 57.5512705
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4751695, upper bound: 57.5513040
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -11.0297852, 46.1416092, -9.0654306, 38.2294083, -49.2591934, 55.2070389
1: -14.0756989, 52.2037125, -11.5589437, 43.2975159, -57.3732147, 63.7626572
2: -13.7119274, 52.0922241, -11.3617239, 42.8905373, -56.6024590, 63.4539490
3: -23.8188839, 55.4197121, -19.6654778, 46.1984787, -70.0173645, 75.0851898
4: -21.8664837, 53.5777702, -18.3505554, 44.1525192, -66.0190048, 71.9282990

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2063044, upper bound: 57.2985682
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4724487, upper bound: 57.4787478
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -10.4046993, 43.4960098, -9.6058359, 40.3157463, -50.7204399, 53.1018333
1: -13.2431650, 49.2144547, -12.2354774, 45.6513290, -58.8944893, 61.4499321
2: -12.9902172, 48.9800606, -12.0278358, 45.2923660, -58.2825851, 61.0078964
3: -22.4255047, 52.3740425, -20.7736835, 48.6870728, -71.1125717, 73.1477203
4: -20.8130035, 50.4081459, -19.3757629, 46.6283188, -67.4413223, 69.7838898

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5019506, upper bound: 57.4934149
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5019506, upper bound: 57.5526292
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -11.0297852, 46.1416092, -10.7050171, 44.6627159, -55.6925011, 56.8466263
1: -14.0756989, 52.2037125, -13.6141796, 50.5332298, -64.6089325, 65.8178940
2: -13.7119274, 52.0922241, -13.3514185, 50.3138771, -64.0258026, 65.4436417
3: -23.8188839, 55.4197121, -23.0363426, 53.7460365, -77.5649185, 78.4560547
4: -21.8664837, 53.5777702, -21.3794994, 51.7781792, -73.6446609, 74.9572678

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -57.1066427, upper bound: 57.0691398
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4724487, upper bound: 57.4787478
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -10.4046993, 43.4960098, -11.3788996, 47.2034073, -57.6081009, 54.8749084
1: -13.2431650, 49.2144547, -14.4609737, 53.3978958, -66.6410599, 63.6754303
2: -12.9902172, 48.9800606, -14.1775379, 53.2335358, -66.2237473, 63.1576004
3: -22.4255047, 52.3740425, -24.4117718, 56.7669830, -79.1924896, 76.7858124
4: -20.8130035, 50.4081459, -22.6555748, 54.8203659, -75.6333618, 73.0636902

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4856191, upper bound: 57.4753677
time: 0.53 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4856191, upper bound: 57.5619695
time: 0.58 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.30 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 0, lower bound: -57.2985682, upper bound: 57.2063044
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 0, lower bound: -57.2985682, upper bound: 57.2063044
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 0, lower bound: -57.4787478, upper bound: 57.4724487
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 0, lower bound: -57.4787478, upper bound: 57.4724487
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 0, lower bound: -57.4751695, upper bound: 57.5019506
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 0, lower bound: -57.4751695, upper bound: 57.5019506
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 0, lower bound: -57.4751695, upper bound: 57.5512705
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 0, lower bound: -57.4751695, upper bound: 57.5513040
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 0, lower bound: -57.2063044, upper bound: 57.2985682
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 0, lower bound: -57.4724487, upper bound: 57.4787478
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 0, lower bound: -57.5019506, upper bound: 57.4934149
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 0, lower bound: -57.5019506, upper bound: 57.5526292
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.30
Output dim: 0, lower bound: -57.1066427, upper bound: 57.0691398
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 0, lower bound: -57.4724487, upper bound: 57.4787478
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 0, lower bound: -57.4856191, upper bound: 57.4753677
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 0, lower bound: -57.4856191, upper bound: 57.5619695

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -8.1608181, 35.4994659, -7.7041636, 33.3772621, -41.5380783, 43.2036285
1: -10.4707947, 40.2127953, -9.8689156, 37.8365364, -48.3073273, 50.0817108
2: -10.2669220, 39.7491150, -9.6831846, 37.3034477, -47.5703697, 49.4323006
3: -18.0829659, 42.9250832, -16.9867744, 40.3695908, -58.4525566, 59.9118462
4: -16.8136044, 40.9275131, -15.7625675, 38.4020462, -55.2156525, 56.6900749

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2985682, upper bound: 57.2063044
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2985682, upper bound: 57.2063044
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -8.1608181, 35.4994659, -9.5042124, 40.4378395, -48.5986557, 45.0036697
1: -10.4707947, 40.2127953, -12.1506071, 45.7674294, -56.2382240, 52.3633995
2: -10.2669220, 39.7491150, -11.8613510, 45.4884071, -55.7553291, 51.6104622
3: -18.0829659, 42.9250832, -20.7276402, 48.6653786, -66.7483444, 63.6527214
4: -16.8136044, 40.9275131, -19.0853729, 46.7810745, -63.5946808, 60.0128860

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2985682, upper bound: 57.2063044
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2985682, upper bound: 57.2063044
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.7259798, 37.0416527, -9.2542639, 39.1532784, -47.8792534, 46.2959175
1: -11.1310406, 41.9613190, -11.8342543, 44.3349075, -55.4659500, 53.7955704
2: -10.9537401, 41.5014725, -11.5614996, 43.9911575, -54.9448929, 53.0629730
3: -18.9851074, 44.7977524, -20.1646118, 47.1800842, -66.1651917, 64.9623642
4: -17.7529068, 42.7079048, -18.5631351, 45.3146896, -63.0675964, 61.2710419

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4005814, upper bound: 57.4347801
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4620306, upper bound: 57.4560967
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.7259798, 37.0416527, -11.0220737, 46.1120911, -54.8380661, 48.0637245
1: -11.1310406, 41.9613190, -14.0660973, 52.1702957, -63.3013382, 56.0274162
2: -10.9537401, 41.5014725, -13.7025061, 52.0582924, -63.0120316, 55.2039795
3: -18.9851074, 44.7977524, -23.8031979, 55.3843269, -74.3694305, 68.6009521
4: -17.7529068, 42.7079048, -21.8516064, 53.5430603, -71.2959671, 64.5595093

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4005814, upper bound: 57.4347801
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4620306, upper bound: 57.4560967
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.1575928, 38.8744278, -8.7464733, 37.1027641, -46.2603569, 47.6209030
1: -11.6959438, 44.0212021, -11.1623430, 42.0329132, -53.7288589, 55.1835442
2: -11.4683819, 43.6604576, -10.9871893, 41.5945511, -53.0629349, 54.6476364
3: -19.9659348, 46.8748016, -19.0206509, 44.8774719, -64.8433990, 65.8954468
4: -18.4506874, 44.9865723, -17.7598114, 42.8391037, -61.2897911, 62.7463837

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4649521, upper bound: 57.4012324
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4926099, upper bound: 57.5013595
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.1575928, 38.8744278, -10.3531132, 43.3094864, -52.4670792, 49.2275352
1: -11.6959438, 44.0212021, -13.1760807, 49.0033646, -60.6993103, 57.1972771
2: -11.4683819, 43.6604576, -12.9312038, 48.7644997, -60.2328796, 56.5916519
3: -19.9659348, 46.8748016, -22.3199158, 52.1524925, -72.1184235, 69.1947174
4: -18.4506874, 44.9865723, -20.7228279, 50.1905785, -68.6412659, 65.7093964

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3868784, upper bound: 57.4012324
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4926099, upper bound: 57.5013595
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.8118200, 37.3577957, -8.7464733, 37.1027641, -45.9145851, 46.1042709
1: -11.2442751, 42.3229713, -11.1623430, 42.0329132, -53.2771873, 53.4853134
2: -11.0679474, 41.8824844, -10.9871893, 41.5945511, -52.6624908, 52.8696671
3: -19.1606712, 45.1869125, -19.0206509, 44.8774719, -64.0381393, 64.2075577
4: -17.8871288, 43.1344337, -17.7598114, 42.8391037, -60.7262230, 60.8942375

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4111822, upper bound: 57.3421243
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4748945, upper bound: 57.5497019
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.8118200, 37.3577957, -10.3531132, 43.3094864, -52.1213074, 47.7109070
1: -11.2442751, 42.3229713, -13.1760807, 49.0033646, -60.2476349, 55.4990501
2: -11.0679474, 41.8824844, -12.9312038, 48.7644997, -59.8324471, 54.8136826
3: -19.1606712, 45.1869125, -22.3199158, 52.1524925, -71.3131638, 67.5068207
4: -17.8871288, 43.1344337, -20.7228279, 50.1905785, -68.0777054, 63.8572617

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4111822, upper bound: 57.3421243
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4748945, upper bound: 57.5497018
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -9.5068970, 40.4510880, -8.1608181, 35.4994659, -45.0063515, 48.6119080
1: -12.1538067, 45.7824173, -10.4707947, 40.2127953, -52.3666000, 56.2532082
2: -11.8646278, 45.5033112, -10.2669220, 39.7491150, -51.6137428, 55.7702332
3: -20.7333374, 48.6806107, -18.0829659, 42.9250832, -63.6584015, 66.7635727
4: -19.0902405, 46.7958794, -16.8136044, 40.9275131, -60.0177536, 63.6094818

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -57.0928336, upper bound: 57.1981825
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.0928336, upper bound: 57.2985682
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -11.0258484, 46.1271629, -8.7259798, 37.0416527, -48.0674973, 54.8531418
1: -14.0707493, 52.1873932, -11.1310406, 41.9613190, -56.0320663, 63.3184319
2: -13.7071276, 52.0755043, -10.9537401, 41.5014725, -55.2085991, 63.0292435
3: -23.8109131, 55.4025574, -18.9851074, 44.7977524, -68.6086655, 74.3876495
4: -21.8592167, 53.5605240, -17.7529068, 42.7079048, -64.5671234, 71.3134308

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -57.0947523, upper bound: 57.2019309
time: 1.03 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.0947523, upper bound: 57.4787478
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -10.4046993, 43.4960098, -9.1575928, 38.8744278, -49.2791176, 52.6535950
1: -13.2431650, 49.2144547, -11.6959438, 44.0212021, -57.2643661, 60.9104004
2: -12.9902172, 48.9800606, -11.4683819, 43.6604576, -56.6506729, 60.4484406
3: -22.4255047, 52.3740425, -19.9659348, 46.8748016, -69.3003006, 72.3399658
4: -20.8130035, 50.4081459, -18.4506874, 44.9865723, -65.7995758, 68.8588257

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2274783, upper bound: 57.2972521
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5013595, upper bound: 57.4926099
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -10.4046993, 43.4960098, -8.8118200, 37.3577957, -47.7624893, 52.3078308
1: -13.2431650, 49.2144547, -11.2442751, 42.3229713, -55.5661354, 60.4587288
2: -12.9902172, 48.9800606, -11.0679474, 41.8824844, -54.8726997, 60.0480080
3: -22.4255047, 52.3740425, -19.1606712, 45.1869125, -67.6124115, 71.5347061
4: -20.8130035, 50.4081459, -17.8871288, 43.1344337, -63.9474335, 68.2952499

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2274783, upper bound: 57.3486191
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5013595, upper bound: 57.5513424
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -11.0258484, 46.1271629, -10.3270845, 43.3149910, -54.3408394, 56.4542465
1: -14.0707493, 52.1873932, -13.1386337, 49.0118065, -63.0825577, 65.3260269
2: -13.7071276, 52.0755043, -12.8930655, 48.7446251, -62.4517517, 64.9685593
3: -23.8109131, 55.4025574, -22.2789650, 52.1449547, -75.9558716, 77.6814880
4: -21.8592167, 53.5605240, -20.6938725, 50.1387177, -71.9979324, 74.2543945

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -57.0884373, upper bound: 57.1915370
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.0884373, upper bound: 57.4787478
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -10.4046993, 43.4960098, -11.0297852, 46.1416092, -56.5463028, 54.5257950
1: -13.2431650, 49.2144547, -14.0756989, 52.2037125, -65.4468765, 63.2901535
2: -12.9902172, 48.9800606, -13.7119274, 52.0922241, -65.0824432, 62.6919861
3: -22.4255047, 52.3740425, -23.8188839, 55.4197121, -77.8452148, 76.1929245
4: -20.8130035, 50.4081459, -21.8664837, 53.5777702, -74.3907700, 72.2746277

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -57.0691398, upper bound: 57.1066427
time: 0.86 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4852651, upper bound: 57.4751227
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -10.4046993, 43.4960098, -10.4046993, 43.4960098, -53.9007034, 53.9006996
1: -13.2431650, 49.2144547, -13.2431650, 49.2144547, -62.4576187, 62.4576187
2: -12.9902172, 48.9800606, -12.9902172, 48.9800606, -61.9702759, 61.9702644
3: -22.4255047, 52.3740425, -22.4255047, 52.3740425, -74.7995453, 74.7995453
4: -20.8130035, 50.4081459, -20.8130035, 50.4081459, -71.2211304, 71.2211304

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.0691398, upper bound: 57.3035365
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4852651, upper bound: 57.5618736
time: 0.62 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.56 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -57.2985682, upper bound: 57.2063044
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -57.2985682, upper bound: 57.2063044
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -57.2985682, upper bound: 57.2063044
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -57.2985682, upper bound: 57.2063044
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -57.4005814, upper bound: 57.4347801
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -57.4620306, upper bound: 57.4560967
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -57.4005814, upper bound: 57.4347801
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -57.4620306, upper bound: 57.4560967
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -57.4649521, upper bound: 57.4012324
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -57.4926099, upper bound: 57.5013595
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -57.3868784, upper bound: 57.4012324
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -57.4926099, upper bound: 57.5013595
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -57.4111822, upper bound: 57.3421243
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -57.4748945, upper bound: 57.5497019
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -57.4111822, upper bound: 57.3421243
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -57.4748945, upper bound: 57.5497018
IS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.56
Output dim: 0, lower bound: -57.0928336, upper bound: 57.1981825
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -57.0928336, upper bound: 57.2985682
IS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.56
Output dim: 0, lower bound: -57.0947523, upper bound: 57.2019309
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -57.0947523, upper bound: 57.4787478
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -57.2274783, upper bound: 57.2972521
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -57.5013595, upper bound: 57.4926099
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -57.2274783, upper bound: 57.3486191
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -57.5013595, upper bound: 57.5513424
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.56
Output dim: 0, lower bound: -57.0884373, upper bound: 57.1915370
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -57.0884373, upper bound: 57.4787478
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.56
Output dim: 0, lower bound: -57.0691398, upper bound: 57.1066427
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -57.4852651, upper bound: 57.4751227
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -57.0691398, upper bound: 57.3035365
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -57.4852651, upper bound: 57.5618736

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -8.6716461, 37.7611389, -7.7041636, 33.3772621, -42.0489082, 45.4652977
1: -11.1499243, 42.7709465, -9.8689156, 37.8365364, -48.9864616, 52.6398621
2: -10.8831701, 42.3993225, -9.6831846, 37.3034477, -48.1866188, 52.0825081
3: -19.2832890, 45.5401001, -16.9867744, 40.3695908, -59.6528778, 62.5268745
4: -17.7571411, 43.6625214, -15.7625675, 38.4020462, -56.1591873, 59.4250870

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4789958, upper bound: 57.4394751
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5004528, upper bound: 57.4981349
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8.0646629, 35.2838440, -7.7041636, 33.3772621, -41.4419212, 42.9880066
1: -10.3480911, 39.9815979, -9.8689156, 37.8365364, -48.1846275, 49.8505135
2: -10.1572323, 39.4773331, -9.6831846, 37.3034477, -47.4606781, 49.1605186
3: -17.9082813, 42.6824608, -16.9867744, 40.3695908, -58.2778702, 59.6692352
4: -16.6487427, 40.6530380, -15.7625675, 38.4020462, -55.0507889, 56.4155998

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4789958, upper bound: 57.4406313
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5004528, upper bound: 57.4994232
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -8.6716461, 37.7611389, -9.5042124, 40.4378395, -49.1094818, 47.2653389
1: -11.1499243, 42.7709465, -12.1506071, 45.7674294, -56.9173546, 54.9215508
2: -10.8831701, 42.3993225, -11.8613510, 45.4884071, -56.3715782, 54.2606735
3: -19.2832890, 45.5401001, -20.7276402, 48.6653786, -67.9486694, 66.2677383
4: -17.7571411, 43.6625214, -19.0853729, 46.7810745, -64.5382156, 62.7478943

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -57.1906359, upper bound: 57.0896194
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -57.1906359, upper bound: 57.2063044
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.0646629, 35.2838440, -9.5042124, 40.4378395, -48.5024986, 44.7880516
1: -10.3480911, 39.9815979, -12.1506071, 45.7674294, -56.1155205, 52.1322060
2: -10.1572323, 39.4773331, -11.8613510, 45.4884071, -55.6456375, 51.3386841
3: -17.9082813, 42.6824608, -20.7276402, 48.6653786, -66.5736618, 63.4101028
4: -16.6487427, 40.6530380, -19.0853729, 46.7810745, -63.4298172, 59.7384109

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -57.1906359, upper bound: 57.0928336
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -57.1906359, upper bound: 57.2063044
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -5.8556199, 27.0330276, -9.2542639, 39.1532784, -45.0088959, 36.2872925
1: -7.4514709, 30.7790146, -11.8342543, 44.3349075, -51.7863770, 42.6132698
2: -7.5214276, 29.8745461, -11.5614996, 43.9911575, -51.5125847, 41.4360466
3: -13.0995874, 33.0493011, -20.1646118, 47.1800842, -60.2796707, 53.2139130
4: -12.6468258, 30.6005001, -18.5631351, 45.3146896, -57.9615135, 49.1636353

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4617751, upper bound: 57.4836434
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4617751, upper bound: 57.4836434
time: 0.52 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -7.0948095, 31.1692696, -7.5399370, 32.6820946, -39.7769051, 38.7092018
1: -9.0766850, 35.4327736, -9.6594334, 37.0595284, -46.1362152, 45.0922089
2: -9.0157633, 34.6674385, -9.4965630, 36.5392342, -45.5549927, 44.1640015
3: -15.5912876, 38.0263023, -16.6003113, 39.5566711, -55.1479492, 54.6266060
4: -14.7595482, 35.6304626, -15.3902140, 37.6483727, -52.4079132, 51.0206757

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5110140, upper bound: 57.5001437
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5110140, upper bound: 57.5001437
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -5.8556199, 27.0330276, -11.0220737, 46.1120911, -51.9677086, 38.0550995
1: -7.4514709, 30.7790146, -14.0660973, 52.1702957, -59.6217651, 44.8451118
2: -7.5214276, 29.8745461, -13.7025061, 52.0582924, -59.5797195, 43.5770531
3: -13.0995874, 33.0493011, -23.8031979, 55.3843269, -68.4839020, 56.8525009
4: -12.6468258, 30.6005001, -21.8516064, 53.5430603, -66.1898880, 52.4520988

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3864468, upper bound: 57.4213149
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3848161, upper bound: 57.3950462
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -7.0948095, 31.1692696, -9.1067457, 38.8031693, -45.8979759, 40.2760010
1: -9.0766850, 35.4327736, -11.6358538, 43.9175301, -52.9942169, 47.0686226
2: -9.0157633, 34.6674385, -11.3853798, 43.6452293, -52.6609917, 46.0528183
3: -15.5912876, 38.0263023, -19.8150902, 46.7222443, -62.3135300, 57.8413925
4: -14.7595482, 35.6304626, -18.2352581, 44.9076538, -59.6672020, 53.8657227

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4475735, upper bound: 57.4424457
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4426336, upper bound: 57.4150010
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -8.7669353, 38.1990242, -7.2883644, 31.7002182, -40.4671440, 45.4873886
1: -11.2745037, 43.2634735, -9.3017883, 35.9763184, -47.2508125, 52.5652618
2: -11.0006275, 42.8983345, -9.2186871, 35.3239288, -46.3245544, 52.1170197
3: -19.5009098, 46.0489769, -16.0208054, 38.5305252, -58.0314331, 62.0697823
4: -17.9422951, 44.1722069, -15.1278973, 36.3534355, -54.2957306, 59.3000984

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4861377, upper bound: 57.4629511
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5058351, upper bound: 57.5123631
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8.8402529, 37.7630539, -8.7431898, 37.0915108, -45.9317627, 46.5062447
1: -11.2974596, 42.7681274, -11.1582193, 42.0202751, -53.3177338, 53.9263458
2: -11.0862007, 42.3602257, -10.9832525, 41.5813866, -52.6675873, 53.3434753
3: -19.3324356, 45.5596771, -19.0141029, 44.8642082, -64.1966400, 64.5737762
4: -17.8893318, 43.6359253, -17.7540627, 42.8253937, -60.7147064, 61.3899727

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5134994, upper bound: 57.5208845
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5134994, upper bound: 57.5238898
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -8.7669353, 38.1990242, -8.9116421, 37.9319496, -46.6988754, 47.1106644
1: -11.2745037, 43.2634735, -11.3547802, 42.9415054, -54.2160034, 54.6182518
2: -11.0006275, 42.8983345, -11.1815090, 42.5283508, -53.5289764, 54.0798378
3: -19.5009098, 46.0489769, -19.3875504, 45.7848320, -65.2857437, 65.4365234
4: -17.9422951, 44.1722069, -18.0740643, 43.7584381, -61.7007332, 62.2462502

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4430553, upper bound: 57.3692448
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2972521, upper bound: 57.2274783
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2972521, upper bound: 57.4012324
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.8402529, 37.7630539, -10.3490896, 43.2948265, -52.1350746, 48.1121445
1: -11.2974596, 42.7681274, -13.1710339, 48.9867973, -60.2842560, 55.9391632
2: -11.0862007, 42.3602257, -12.9263172, 48.7475052, -59.8337059, 55.2865448
3: -19.3324356, 45.5596771, -22.3117828, 52.1350784, -71.4675140, 67.8714600
4: -17.8893318, 43.6359253, -20.7154007, 50.1729965, -68.0623169, 64.3513184

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2972521, upper bound: 57.2274783
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2972521, upper bound: 57.5013595
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -8.0646629, 35.2838440, -7.2883644, 31.7002182, -39.7648735, 42.5722084
1: -10.3480911, 39.9815979, -9.3017883, 35.9763184, -46.3244095, 49.2833862
2: -10.1572323, 39.4773331, -9.2186871, 35.3239288, -45.4811630, 48.6960220
3: -17.9082813, 42.6824608, -16.0208054, 38.5305252, -56.4388046, 58.7032661
4: -16.6487427, 40.6530380, -15.1278973, 36.3534355, -53.0021782, 55.7809372

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4962776, upper bound: 57.4639298
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5068292, upper bound: 57.5136952
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -8.4693804, 36.1834831, -8.7431898, 37.0915108, -45.5608902, 44.9266739
1: -10.8127460, 41.0070839, -11.1582193, 42.0202751, -52.8330231, 52.1653023
2: -10.6570034, 40.5084305, -10.9832525, 41.5813866, -52.2383881, 51.4916840
3: -18.4758568, 43.8035660, -19.0141029, 44.8642082, -63.3400650, 62.8176689
4: -17.2884102, 41.7039909, -17.7540627, 42.8253937, -60.1138000, 59.4580498

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5087887, upper bound: 57.5299729
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.5351403, upper bound: 57.5497019
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -8.0646629, 35.2838440, -8.9116421, 37.9319496, -45.9966087, 44.1954880
1: -10.3480911, 39.9815979, -11.3547802, 42.9415054, -53.2895966, 51.3363724
2: -10.1572323, 39.4773331, -11.1815090, 42.5283508, -52.6855850, 50.6588364
3: -17.9082813, 42.6824608, -19.3875504, 45.7848320, -63.6931152, 62.0700111
4: -16.6487427, 40.6530380, -18.0740643, 43.7584381, -60.4071808, 58.7270927

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3750019, upper bound: 57.3029663
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3310183, upper bound: 57.2533148
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3310183, upper bound: 57.3421243
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -8.4693804, 36.1834831, -10.3490896, 43.2948265, -51.7641945, 46.5325737
1: -10.8127460, 41.0070839, -13.1710339, 48.9867973, -59.7995453, 54.1781158
2: -10.6570034, 40.5084305, -12.9263172, 48.7475052, -59.4045067, 53.4347458
3: -18.4758568, 43.8035660, -22.3117828, 52.1350784, -70.6109314, 66.1153488
4: -17.2884102, 41.7039909, -20.7154007, 50.1729965, -67.4614105, 62.4193916

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3549957, upper bound: 57.2743375
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3549957, upper bound: 57.5497019
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -10.6520157, 44.6244659, -8.1608181, 35.4994659, -46.1514816, 52.7852859
1: -13.5943565, 50.4944687, -10.4707947, 40.2127953, -53.8071480, 60.9652557
2: -13.2469320, 50.3344727, -10.2669220, 39.7491150, -52.9960480, 60.6013947
3: -23.0409489, 53.6432571, -18.0829659, 42.9250832, -65.9660263, 71.7262268
4: -21.1693459, 51.7696075, -16.8136044, 40.9275131, -62.0968590, 68.5832138

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.0896194, upper bound: 57.2985682
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.0896194, upper bound: 57.2985682
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -10.6957531, 44.9465866, -8.7259798, 37.0416527, -47.7374001, 53.6725578
1: -13.6549959, 50.8562546, -11.1310406, 41.9613190, -55.6163139, 61.9872971
2: -13.3063459, 50.7025528, -10.9537401, 41.5014725, -54.8078194, 61.6562653
3: -23.1477661, 54.0001144, -18.9851074, 44.7977524, -67.9455185, 72.9852219
4: -21.2595654, 52.1365585, -17.7529068, 42.7079048, -63.9674683, 69.8894501

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.0844645, upper bound: 57.4748945
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.0844645, upper bound: 57.4787478
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -10.0193787, 42.8943863, -7.5722671, 32.9230690, -42.9424477, 50.4666519
1: -12.8247375, 48.5298729, -9.6884995, 37.3267403, -50.1514778, 58.2183723
2: -12.5208769, 48.2823448, -9.5402012, 36.7739372, -49.2948074, 57.8225365
3: -21.9548664, 51.6154327, -16.7127934, 39.8556290, -61.8104935, 68.3282242
4: -20.2769165, 49.6436996, -15.5613279, 37.8688316, -58.1457481, 65.2050247

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2274783, upper bound: 57.2972521
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2274783, upper bound: 57.2972521
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -10.0372343, 42.1927071, -9.1544914, 38.8635445, -48.9007721, 51.3471985
1: -12.7809610, 47.7437553, -11.6920643, 44.0089111, -56.7898712, 59.4358215
2: -12.5463152, 47.4627686, -11.4646530, 43.6477509, -56.1940651, 58.9274216
3: -21.6898041, 50.8244820, -19.9597588, 46.8618927, -68.5516968, 70.7842407
4: -20.1472874, 48.8353882, -18.4451962, 44.9733658, -65.1206436, 67.2805786

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4012324, upper bound: 57.4649521
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4012313, upper bound: 57.4926099
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -10.0193787, 42.8943863, -7.3191862, 31.8214073, -41.8407860, 50.2135735
1: -12.8247375, 48.5298729, -9.3405285, 36.1141624, -48.9389000, 57.8703995
2: -12.5208769, 48.2823448, -9.2568541, 35.4603920, -47.9812622, 57.5391960
3: -21.9548664, 51.6154327, -16.0874615, 38.6773643, -60.6322289, 67.7028961
4: -20.2769165, 49.6436996, -15.1883612, 36.4939537, -56.7708702, 64.8320541

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2712372, upper bound: 57.3462861
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2712372, upper bound: 57.3486192
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -10.0372343, 42.1927071, -8.8084831, 37.3463173, -47.3835487, 51.0011864
1: -12.7809610, 47.7437553, -11.2400799, 42.3101044, -55.0910645, 58.9838333
2: -12.5463152, 47.4627686, -11.0639439, 41.8690529, -54.4153671, 58.5267105
3: -21.6898041, 50.8244820, -19.1540031, 45.1733856, -66.8631744, 69.9784775
4: -20.1472874, 48.8353882, -17.8812752, 43.1204681, -63.2677498, 66.7166519

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4631505, upper bound: 57.5031772
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4631505, upper bound: 57.5513424
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -10.6957531, 44.9465866, -10.3270845, 43.3149910, -54.0107422, 55.2736664
1: -13.6549959, 50.8562546, -13.1386337, 49.0118065, -62.6668015, 63.9948883
2: -13.3063459, 50.7025528, -12.8930655, 48.7446251, -62.0509720, 63.5955887
3: -23.1477661, 54.0001144, -22.2789650, 52.1449547, -75.2927246, 76.2790756
4: -21.2595654, 52.1365585, -20.6938725, 50.1387177, -71.3982849, 72.8304291

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -56.9753371, upper bound: 57.4665625
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -56.9753371, upper bound: 57.4787478
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -10.0372343, 42.1927071, -11.0258484, 46.1271629, -56.1643944, 53.2185478
1: -12.7809610, 47.7437553, -14.0707493, 52.1873932, -64.9683456, 61.8145065
2: -12.5463152, 47.4627686, -13.7071276, 52.0755043, -64.6218185, 61.1698952
3: -21.6898041, 50.8244820, -23.8109131, 55.4025574, -77.0923386, 74.6353912
4: -20.1472874, 48.8353882, -21.8592167, 53.5605240, -73.7078094, 70.6945953

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -57.1915370, upper bound: 57.0884373
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.1915370, upper bound: 57.4751227
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -10.0193787, 42.8943863, -8.9207363, 37.9650917, -47.9844704, 51.8151245
1: -12.8247375, 48.5298729, -11.3667240, 42.9788628, -55.8035965, 59.8965988
2: -12.5208769, 48.2823448, -11.1918030, 42.5667305, -55.0876007, 59.4741402
3: -21.9548664, 51.6154327, -19.4062309, 45.8236084, -67.7784729, 71.0216675
4: -20.2769165, 49.6436996, -18.0897503, 43.7970161, -64.0739288, 67.7334518

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
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
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.2045298, upper bound: 57.3035372
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -10.0372343, 42.1927071, -10.4005337, 43.4808159, -53.5180435, 52.5932388
1: -12.7809610, 47.7437553, -13.2379370, 49.1972885, -61.9782372, 60.9816933
2: -12.5463152, 47.4627686, -12.9851618, 48.9624748, -61.5087891, 60.4479294
3: -21.6898041, 50.8244820, -22.4170742, 52.3559875, -74.0457840, 73.2415543
4: -20.1472874, 48.8353882, -20.8053131, 50.3899536, -70.5372314, 69.6407013

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3547935, upper bound: 57.2741648
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3547935, upper bound: 57.5618736
time: 0.67 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.79 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.4789958, upper bound: 57.4394751
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.5004528, upper bound: 57.4981349
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.4789958, upper bound: 57.4406313
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.5004528, upper bound: 57.4994232
IS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.1906359, upper bound: 57.0896194
IS_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.1906359, upper bound: 57.2063044
IS_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.1906359, upper bound: 57.0928336
IS_A1_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.1906359, upper bound: 57.2063044
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.4617751, upper bound: 57.4836434
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.4617751, upper bound: 57.4836434
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.5110140, upper bound: 57.5001437
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.5110140, upper bound: 57.5001437
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.3864468, upper bound: 57.4213149
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.3848161, upper bound: 57.3950462
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.4475735, upper bound: 57.4424457
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.4426336, upper bound: 57.4150010
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.4861377, upper bound: 57.4629511
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.5058351, upper bound: 57.5123631
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.5134994, upper bound: 57.5208845
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.5134994, upper bound: 57.5238898
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.2972521, upper bound: 57.2274783
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.2972521, upper bound: 57.4012324
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.2972521, upper bound: 57.2274783
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.2972521, upper bound: 57.5013595
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.4962776, upper bound: 57.4639298
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.5068292, upper bound: 57.5136952
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.5087887, upper bound: 57.5299729
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.5351403, upper bound: 57.5497019
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.3310183, upper bound: 57.2533148
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.3310183, upper bound: 57.3421243
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.3549957, upper bound: 57.2743375
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.3549957, upper bound: 57.5497019
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.0896194, upper bound: 57.2985682
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.0896194, upper bound: 57.2985682
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.0844645, upper bound: 57.4748945
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.0844645, upper bound: 57.4787478
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.2274783, upper bound: 57.2972521
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.2274783, upper bound: 57.2972521
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.4012324, upper bound: 57.4649521
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.4012313, upper bound: 57.4926099
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.2712372, upper bound: 57.3462861
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.2712372, upper bound: 57.3486192
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.4631505, upper bound: 57.5031772
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.4631505, upper bound: 57.5513424
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -56.9753371, upper bound: 57.4665625
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -56.9753371, upper bound: 57.4787478
IS_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.1915370, upper bound: 57.0884373
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.1915370, upper bound: 57.4751227
IS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.2045298, upper bound: 57.2045298
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.2045298, upper bound: 57.3035372
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.3547935, upper bound: 57.2741648
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.79
Output dim: 0, lower bound: -57.3547935, upper bound: 57.5618736

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -8.6716461, 37.7611389, -4.9572344, 23.8805923, -32.5522270, 42.7183647
1: -11.1499243, 42.7709465, -6.3238821, 27.2633629, -38.4132881, 49.0948296
2: -10.8831701, 42.3993225, -6.4199829, 26.3257694, -37.2089386, 48.8193054
3: -19.2832890, 45.5401001, -11.2875338, 29.1984577, -48.4817390, 56.8276329
4: -17.7571411, 43.6625214, -10.9090214, 26.9078255, -44.6649666, 54.5715408

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4218930, upper bound: 57.4191323
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4218930, upper bound: 57.4394751
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6.7118711, 30.4809456, -5.8895493, 26.9483929, -33.6602631, 36.3704910
1: -8.6158180, 34.5807228, -7.5145993, 30.7020702, -39.3178864, 42.0953217
2: -8.5209246, 33.9804001, -7.5454774, 29.8227329, -38.3436584, 41.5258713
3: -15.1375408, 36.9538155, -13.1030102, 32.9425545, -48.0800896, 50.0568237
4: -14.1485462, 34.9880142, -12.5207777, 30.6023521, -44.7508965, 47.5087891

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4435099, upper bound: 57.4778288
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4435099, upper bound: 57.4981350
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.0646629, 35.2838440, -4.9572344, 23.8805923, -31.9452457, 40.2410774
1: -10.3480911, 39.9815979, -6.3238821, 27.2633629, -37.6114540, 46.3054810
2: -10.1572323, 39.4773331, -6.4199829, 26.3257694, -36.4830017, 45.8973160
3: -17.9082813, 42.6824608, -11.2875338, 29.1984577, -47.1067390, 53.9699936
4: -16.6487427, 40.6530380, -10.9090214, 26.9078255, -43.5565681, 51.5620575

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4348036, upper bound: 57.4231747
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4348036, upper bound: 57.4406313
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6.8749752, 30.8461609, -5.8895493, 26.9483929, -33.8233681, 36.7357101
1: -8.8059387, 35.0082703, -7.5145993, 30.7020702, -39.5080032, 42.5228691
2: -8.7225323, 34.3330879, -7.5454774, 29.8227329, -38.5452652, 41.8785591
3: -15.3818035, 37.4852638, -13.1030102, 32.9425545, -48.3243523, 50.5882721
4: -14.4695282, 35.3545418, -12.5207777, 30.6023521, -45.0718765, 47.8753128

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4577240, upper bound: 57.4822883
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4577240, upper bound: 57.4994233
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -5.8556199, 27.0330276, -8.9457722, 38.7898598, -44.6454773, 35.9787979
1: -7.4514709, 30.7790146, -11.5194845, 43.9305115, -51.3819809, 42.2985001
2: -7.5214276, 29.8745461, -11.1893988, 43.5970116, -51.1184387, 41.0639458
3: -13.0995874, 33.0493011, -19.8603401, 46.7209511, -59.8205376, 52.9096413
4: -12.6468258, 30.6005001, -18.2107544, 44.8687057, -57.5155334, 48.8112450

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4364782, upper bound: 57.4738660
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4364782, upper bound: 57.4836434
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -5.8556199, 27.0330276, -8.9451551, 38.0861778, -43.9417915, 35.9781837
1: -7.4514709, 30.7790146, -11.4452267, 43.1305504, -50.5820198, 42.2242432
2: -7.5214276, 29.8745461, -11.1906691, 42.7414474, -50.2628746, 41.0652161
3: -13.0995874, 33.0493011, -19.5482712, 45.9143333, -59.0139160, 52.5975647
4: -12.6468258, 30.6005001, -18.0214596, 44.0143204, -56.6611481, 48.6219597

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4364782, upper bound: 57.4738660
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4364782, upper bound: 57.4836434
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -7.0948095, 31.1692696, -6.9090242, 31.1947784, -38.2895889, 38.0782890
1: -9.0766850, 35.4327736, -8.8870697, 35.3820496, -44.4587326, 44.3198357
2: -9.0157633, 34.6674385, -8.7317686, 34.8001747, -43.8159332, 43.3992081
3: -15.5912876, 38.0263023, -15.5480213, 37.7371979, -53.3284836, 53.5743217
4: -14.7595482, 35.6304626, -14.4439373, 35.8108368, -50.5703773, 50.0743980

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4947912, upper bound: 57.4947913
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4947912, upper bound: 57.5001437
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -7.0948095, 31.1692696, -7.2260208, 31.6409607, -38.7357674, 38.3952827
1: -9.0766850, 35.4327736, -9.2570448, 35.8910789, -44.9677620, 44.6898193
2: -9.0157633, 34.6674385, -9.1198711, 35.3156853, -44.3314476, 43.7873077
3: -15.5912876, 38.0263023, -15.9647665, 38.3252106, -53.9164925, 53.9910698
4: -14.7595482, 35.6304626, -14.8524237, 36.3636703, -51.1232071, 50.4828873

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4947912, upper bound: 57.4947913
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4947912, upper bound: 57.5001437
time: 0.55 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -5.1743493, 24.5558052, -9.2198343, 39.4563866, -44.6307297, 33.7756386
1: -6.5614610, 28.0330658, -11.8139801, 44.6410828, -51.2025375, 39.8470459
2: -6.7016864, 27.0519428, -11.5357418, 44.4028244, -51.1045074, 38.5876846
3: -11.6108456, 30.0989265, -20.1266613, 47.4571190, -59.0679626, 50.2255821
4: -11.3643789, 27.6201801, -18.4494286, 45.6943665, -57.0587463, 46.0695992

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3841931, upper bound: 57.4205225
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3841931, upper bound: 57.4213149
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -5.3310795, 25.0389805, -10.8337460, 45.4515839, -50.7826538, 35.8727264
1: -6.7702265, 28.5774231, -13.8244686, 51.4227791, -58.1930046, 42.4018936
2: -6.9082913, 27.6134663, -13.5274858, 51.2218475, -58.1301384, 41.1409531
3: -11.9094334, 30.7333107, -23.3656330, 54.6374168, -66.5468521, 54.0989380
4: -11.7223749, 28.1843834, -21.5633430, 52.6907158, -64.4130783, 49.7477264

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3798319, upper bound: 57.3934056
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.3798319, upper bound: 57.3950462
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6.5085258, 29.0295506, -7.6789103, 33.5629921, -40.0715179, 36.7084618
1: -8.3258076, 33.0353851, -9.8367443, 38.0084076, -46.3342133, 42.8721313
2: -8.3090544, 32.1982574, -9.6771469, 37.6000862, -45.9091415, 41.8754044
3: -14.3440247, 35.4950104, -16.8793716, 40.5001755, -54.8442001, 52.3743782
4: -13.6604967, 33.0822220, -15.5708542, 38.7242851, -52.3847694, 48.6530762

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4468693, upper bound: 57.4421306
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4468693, upper bound: 57.4424457
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6.1887064, 28.0192242, -8.9253216, 38.2394524, -44.4281578, 36.9445457
1: -7.9055061, 31.9337635, -11.4145002, 43.2842903, -51.1897964, 43.3482628
2: -7.9516935, 31.0157528, -11.2278595, 42.9302292, -50.8819237, 42.2436142
3: -13.6534290, 34.3535194, -19.4100628, 46.0937347, -59.7471619, 53.7635803
4: -13.1916323, 31.7836227, -17.9797783, 44.1570206, -57.3486481, 49.7634010

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4414349, upper bound: 57.4143986
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4414349, upper bound: 57.4150010
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -8.7669353, 38.1990242, -4.6831083, 22.6340981, -31.4010315, 42.8821335
1: -11.2745037, 43.2634735, -5.9544382, 25.9099998, -37.1844902, 49.2179108
2: -11.0006275, 42.8983345, -6.1071711, 24.8895149, -35.8901443, 49.0055046
3: -19.5009098, 46.0489769, -10.5957079, 27.8443794, -47.3452911, 56.6446838
4: -17.9422951, 44.1722069, -10.4759359, 25.3687973, -43.3110924, 54.6481438

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4218930, upper bound: 57.4410069
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -57.4276039, upper bound: 57.4629511
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6.7118711, 30.4809456, -5.9183612, 26.9759464, -33.6878090, 36.3992958
1: -8.6158180, 34.5807228, -7.5294509, 30.7693691, -39.3851852, 42.1101723
2: -8.5209246, 33.9804001, -7.6058655, 29.8019161, -38.3228416, 41.5862656
3: -15.1375408, 36.9538155, -13.0700216, 33.0934677, -48.2310066, 50.0238380
4: -14.1485462, 34.9880142, -12.6719685, 30.5429916, -44.6915359, 47.6599731

Time for backsubstitution: 2.23 seconds
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=66.57380676269531
rel_dist={0: [-57.5686962838552, 57.5686962838552]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 1126.14 seconds
