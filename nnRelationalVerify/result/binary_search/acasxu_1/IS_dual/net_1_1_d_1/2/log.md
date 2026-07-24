## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_1.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 1.0495482984


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489)
1: (-0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965)
2: (-0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164)
3: (-0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897)
4: (-0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601)

## BASE Result
execution time: IAR + LP analysis = 1.53 + 1.07 = 2.60 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -1.0564887, upper bound: 1.0564887


# Binary Search by BASE starts (time budget: 1197.40 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.1000000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.1000000, mid=0.1000000, abs_max=1.1883488893508911
rel_dist={0: [-1.0558835892060525, 1.0558835892060525]}

## Binary search (step 1) starts
Candidate diff: 0.0500000


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0500000, mid=0.0500000, abs_max=1.1883488893508911
rel_dist={0: [-1.0553818619849304, 1.0553818619849311]}

## Binary search (step 2) starts
Candidate diff: 0.0250000


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0250000, mid=0.0250000, abs_max=1.1883488893508911
rel_dist={0: [-1.0551075589159629, 1.0551075589159629]}

## Binary search (step 3) starts
Candidate diff: 0.0125000


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0125000, mid=0.0125000, abs_max=1.1883488893508911
rel_dist={0: [-1.0549470781280466, 1.0549470781280457]}

## Binary search (step 4) starts
Candidate diff: 0.0062500


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0062500, mid=0.0062500, abs_max=1.1883488893508911
rel_dist={0: [-1.0548482094125333, 1.0548482094125333]}

## Binary search (step 5) starts
Candidate diff: 0.0031250


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0031250, mid=0.0031250, abs_max=1.1883488893508911
rel_dist={0: [-1.0547849699636636, 1.0547849699636629]}

## Binary search (step 6) starts
Candidate diff: 0.0015625


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0015625, mid=0.0015625, abs_max=1.1883488893508911
rel_dist={0: [-1.0547487716980937, 1.054748771698094]}

## Binary search (step 7) starts
Candidate diff: 0.0007812


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0007812, mid=0.0007812, abs_max=1.1883488893508911
rel_dist={0: [-1.054717313123089, 1.0547173131230885]}

## Binary search (step 8) starts
Candidate diff: 0.0003906


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0003906, mid=0.0003906, abs_max=1.1883488893508911
rel_dist={0: [-1.054693067218044, 1.0546930672156352]}

## Binary search (step 9) starts
Candidate diff: 0.0001953


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0001953, mid=0.0001953, abs_max=1.1883488893508911
rel_dist={0: [-1.0546768688586334, 1.0546768688579005]}

## Binary search (step 10) starts
Candidate diff: 0.0000977


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0000977, mid=0.0000977, abs_max=1.1883488893508911
rel_dist={0: [-1.0546645489281865, 1.0546645489275417]}

## Binary search (step 11) starts
Candidate diff: 0.0000488


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0000488, mid=0.0000488, abs_max=1.1883488893508911
rel_dist={0: [-1.0546556231562683, 1.0546556237213354]}

## Binary search (step 12) starts
Candidate diff: 0.0000244


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0000244, mid=0.0000244, abs_max=1.1883488893508911
rel_dist={0: [-1.0546509986945563, 1.0546509989519355]}

## Binary search (step 13) starts
Candidate diff: 0.0000122


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000122, mid=0.0000122, abs_max=1.1883488893508911
rel_dist={0: [-1.0546486318282617, 1.0546486319569248]}

## Binary search (step 14) starts
Candidate diff: 0.0000061


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000061, mid=0.0000061, abs_max=1.1883488893508911
rel_dist={0: [-1.0546474519922995, 1.054647448669685]}

## Binary search (step 15) starts
Candidate diff: 0.0000031


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000031, mid=0.0000031, abs_max=1.1883488893508911
rel_dist={0: [-1.054646874919929, 1.0546468574895504]}

## Binary search (step 16) starts
Candidate diff: 0.0000015


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000015, mid=0.0000015, abs_max=1.1883488893508911
rel_dist={0: [-1.0546465944949381, 1.0546465655883437]}

## Binary search (step 17) starts
Candidate diff: 0.0000008


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000008, mid=0.0000008, abs_max=1.1883488893508911
rel_dist={0: [-1.05464671617786, 1.0546464864943261]}

## Binary Search Result
Binary search time: 46.61 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 1150.79 seconds

## Binary search (step 0) starts
Candidate diff: 0.1000000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0553496, upper bound: 1.0511832
time: 0.33 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
time: 0.34 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.82 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.82
Output dim: 0, lower bound: -1.0553496, upper bound: 1.0511832
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.82
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.2352601, 0.7538407, -0.3035725, 0.8847764, -1.1200366, 1.0574131
1: -0.4706453, 0.9488738, -0.5660125, 1.0933844, -1.5640295, 1.5148864
2: -0.3915833, 1.0723588, -0.4826685, 1.2412479, -1.6328310, 1.5550274
3: -0.8320177, 1.0878556, -0.9617165, 1.2755736, -2.1075912, 2.0495720
4: -0.7029035, 1.3031529, -0.8354526, 1.4994075, -2.2023110, 2.1386056

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
time: 0.35 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
time: 0.37 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -1.3271873, 2.7002649, -0.3035725, 0.8847764, -2.2119637, 3.0038373
1: -1.8266034, 2.9257355, -0.5660125, 1.0933844, -2.9199877, 3.4917479
2: -1.7866864, 3.3100519, -0.4826685, 1.2412479, -3.0279343, 3.7927201
3: -2.3538351, 3.8103127, -0.9617165, 1.2755736, -3.6294079, 4.7720289
4: -2.7129741, 3.8588223, -0.8354526, 1.4994075, -4.2123814, 4.6942744

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
time: 0.34 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
time: 0.35 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.32 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.32
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.32
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.32
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.32
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.2352601, 0.7538407, -0.2352601, 0.7538407, -0.9891008, 0.9891006
1: -0.4706453, 0.9488738, -0.4706453, 0.9488738, -1.4195192, 1.4195192
2: -0.3915833, 1.0723588, -0.3915833, 1.0723588, -1.4639422, 1.4639422
3: -0.8320177, 1.0878556, -0.8320177, 1.0878556, -1.9198732, 1.9198732
4: -0.7029035, 1.3031529, -0.7029035, 1.3031529, -2.0060563, 2.0060563

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0549353, upper bound: 1.0510210
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535985, upper bound: 1.0510581
time: 0.36 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.2352601, 0.7538407, -1.3271873, 2.7002649, -2.9355247, 2.0810280
1: -0.4706453, 0.9488738, -1.8266034, 2.9257355, -3.3963809, 2.7754772
2: -0.3915833, 1.0723588, -1.7866864, 3.3100519, -3.7016354, 2.8590453
3: -0.8320177, 1.0878556, -2.3538351, 3.8103127, -4.6423302, 3.4416907
4: -0.7029035, 1.3031529, -2.7129741, 3.8588223, -4.5617256, 4.0161266

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0553496, upper bound: 1.0511352
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0522625, upper bound: 1.0507646
time: 0.40 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -1.3271873, 2.7002649, -0.2352601, 0.7538407, -2.0810280, 2.9355249
1: -1.8266034, 2.9257355, -0.4706453, 0.9488738, -2.7754772, 3.3963809
2: -1.7866864, 3.3100519, -0.3915833, 1.0723588, -2.8590453, 3.7016351
3: -2.3538351, 3.8103127, -0.8320177, 1.0878556, -3.4416907, 4.6423302
4: -2.7129741, 3.8588223, -0.7029035, 1.3031529, -4.0161266, 4.5617256

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507166, upper bound: 1.0507511
time: 0.37 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507030, upper bound: 1.0507030
time: 0.42 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -1.3271873, 2.7002649, -1.3271873, 2.7002649, -4.0274525, 4.0274525
1: -1.8266034, 2.9257355, -1.8266034, 2.9257355, -4.7523389, 4.7523384
2: -1.7866864, 3.3100519, -1.7866864, 3.3100519, -5.0967383, 5.0967379
3: -2.3538351, 3.8103127, -2.3538351, 3.8103127, -6.1641479, 6.1641479
4: -2.7129741, 3.8588223, -2.7129741, 3.8588223, -6.5717964, 6.5717964

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507166, upper bound: 1.0507511
time: 0.37 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507030, upper bound: 1.0507030
time: 0.41 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.39 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 0, lower bound: -1.0549353, upper bound: 1.0510210
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 0, lower bound: -1.0535985, upper bound: 1.0510581
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 0, lower bound: -1.0553496, upper bound: 1.0511352
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 0, lower bound: -1.0522625, upper bound: 1.0507646
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 0, lower bound: -1.0507166, upper bound: 1.0507511
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 0, lower bound: -1.0507030, upper bound: 1.0507030
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 0, lower bound: -1.0507166, upper bound: 1.0507511
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 0, lower bound: -1.0507030, upper bound: 1.0507030

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.2013956, 0.7023641, -0.2352601, 0.7538407, -0.9552363, 0.9376242
1: -0.4206367, 0.8831170, -0.4706453, 0.9488738, -1.3695104, 1.3537623
2: -0.3466128, 1.0003821, -0.3915833, 1.0723588, -1.4189715, 1.3919654
3: -0.7602279, 1.0080743, -0.8320177, 1.0878556, -1.8480835, 1.8400919
4: -0.6370696, 1.2147777, -0.7029035, 1.3031529, -1.9402225, 1.9176812

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0552379, upper bound: 1.0518924
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A1_A2

### Relational analysis result of IS_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508607, upper bound: 1.0508939
time: 0.33 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.2363607, 0.7416377, -0.2328624, 0.7488655, -0.9852261, 0.9745001
1: -0.4722191, 0.9222932, -0.4671082, 0.9423364, -1.4145554, 1.3894014
2: -0.3964722, 1.0577438, -0.3885702, 1.0657597, -1.4622319, 1.4463140
3: -0.8244337, 1.0810699, -0.8270060, 1.0809959, -1.9054296, 1.9080759
4: -0.7141775, 1.3009543, -0.6987019, 1.2962005, -2.0103781, 1.9996562

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0536465, upper bound: 1.0519771
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0516066, upper bound: 1.0516066
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.2121267, 0.7201707, -1.3271873, 2.7002649, -2.9123917, 2.0473580
1: -0.4377390, 0.9047500, -1.8266034, 2.9257355, -3.3634744, 2.7313535
2: -0.3605663, 1.0264064, -1.7866864, 3.3100519, -3.6706183, 2.8130927
3: -0.7864621, 1.0343318, -2.3538351, 3.8103127, -4.5967746, 3.3881669
4: -0.6582627, 1.2474420, -2.7129741, 3.8588223, -4.5170841, 3.9604161

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0489561, upper bound: 1.0405855
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0549353, upper bound: 1.0509748
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535849, upper bound: 1.0510412
time: 0.33 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.2998355, 0.9004637, -1.3076617, 2.6589031, -2.9587383, 2.2081251
1: -0.5571232, 1.1422836, -1.8009224, 2.8803883, -3.4375114, 2.9432056
2: -0.4746163, 1.2865998, -1.7620664, 3.2624674, -3.7370837, 3.0486660
3: -0.9696201, 1.2997241, -2.3227785, 3.7524827, -4.7221026, 3.6225016
4: -0.8469371, 1.4981039, -2.6778111, 3.8049459, -4.6518831, 4.1759148

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507602, upper bound: 1.0497692
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0515450, upper bound: 1.0506707
time: 0.49 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -1.3271873, 2.7002649, -0.2121267, 0.7201707, -2.0473580, 2.9123917
1: -1.8266034, 2.9257355, -0.4377390, 0.9047500, -2.7313535, 3.3634744
2: -1.7866864, 3.3100519, -0.3605663, 1.0264064, -2.8130927, 3.6706183
3: -2.3538351, 3.8103127, -0.7864621, 1.0343318, -3.3881669, 4.5967746
4: -2.7129741, 3.8588223, -0.6582627, 1.2474420, -3.9604161, 4.5170846

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0405855, upper bound: 1.0489561
time: 0.39 seconds

## Relational analysis of IS_A2_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0509748, upper bound: 1.0549353
time: 0.37 seconds

## Relational analysis of IS_A2_B1_B1_B2

### Relational analysis result of IS_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0510412, upper bound: 1.0535849
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -1.3076617, 2.6589031, -0.2998355, 0.9004637, -2.2081251, 2.9587383
1: -1.8009224, 2.8803883, -0.5571232, 1.1422836, -2.9432056, 3.4375114
2: -1.7620664, 3.2624674, -0.4746163, 1.2865998, -3.0486655, 3.7370837
3: -2.3227785, 3.7524827, -0.9696201, 1.2997241, -3.6225021, 4.7221026
4: -2.6778111, 3.8049459, -0.8469371, 1.4981039, -4.1759148, 4.6518822

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0497692, upper bound: 1.0507602
time: 0.42 seconds

## Relational analysis of IS_A2_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506707, upper bound: 1.0515450
time: 0.39 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -1.3271873, 2.7002649, -1.3108072, 2.6636837, -3.9908710, 4.0110712
1: -1.8266034, 2.9257355, -1.8037789, 2.8828609, -4.7094641, 4.7295141
2: -1.7866864, 3.3100519, -1.7677765, 3.2624910, -5.0491772, 5.0778284
3: -2.3538351, 3.8103127, -2.3203149, 3.7582724, -6.1121068, 6.1306272
4: -2.7129741, 3.8588223, -2.6851826, 3.8056340, -6.5186081, 6.5440044

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_B1_B1

### Relational analysis result of IS_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0491613, upper bound: 1.0496032
time: 0.37 seconds

## Relational analysis of IS_A2_B2_B1_B2

### Relational analysis result of IS_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506227, upper bound: 1.0506259
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -1.3076617, 2.6589031, -1.3019793, 2.7201226, -4.0277839, 3.9608824
1: -1.8009224, 2.8803883, -1.7827781, 2.9920893, -4.7930117, 4.6631665
2: -1.7620664, 3.2624674, -1.7184958, 3.3870904, -5.1491566, 4.9809632
3: -2.3227785, 3.7524827, -2.3666272, 3.8082108, -6.1309886, 6.1191101
4: -2.6778111, 3.8049459, -2.6377411, 3.8847511, -6.5625620, 6.4426870

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_B2_B1

### Relational analysis result of IS_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0488467, upper bound: 1.0486747
time: 0.36 seconds

## Relational analysis of IS_A2_B2_B2_B2

### Relational analysis result of IS_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506091, upper bound: 1.0506091
time: 0.42 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.54 seconds
IS_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 2.54
Output dim: 0, lower bound: -1.0552379, upper bound: 1.0518924
IS_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 2.54
Output dim: 0, lower bound: -1.0508607, upper bound: 1.0508939
IS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 2.54
Output dim: 0, lower bound: -1.0536465, upper bound: 1.0519771
IS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 2.54
Output dim: 0, lower bound: -1.0516066, upper bound: 1.0516066
IS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 2.54
Output dim: 0, lower bound: -1.0549353, upper bound: 1.0509748
IS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 2.54
Output dim: 0, lower bound: -1.0535849, upper bound: 1.0510412
IS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 2.54
Output dim: 0, lower bound: -1.0507602, upper bound: 1.0497692
IS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 2.54
Output dim: 0, lower bound: -1.0515450, upper bound: 1.0506707
IS_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.54
Output dim: 0, lower bound: -1.0509748, upper bound: 1.0549353
IS_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.54
Output dim: 0, lower bound: -1.0510412, upper bound: 1.0535849
IS_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.54
Output dim: 0, lower bound: -1.0497692, upper bound: 1.0507602
IS_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.54
Output dim: 0, lower bound: -1.0506707, upper bound: 1.0515450
IS_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.54
Output dim: 0, lower bound: -1.0491613, upper bound: 1.0496032
IS_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.54
Output dim: 0, lower bound: -1.0506227, upper bound: 1.0506259
IS_A2_B2_B2_B1, status: Status.VERIFIED, split count: 4, time: 2.54
Output dim: 0, lower bound: -1.0488467, upper bound: 1.0486747
IS_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.54
Output dim: 0, lower bound: -1.0506091, upper bound: 1.0506091

## BFS IS instance: IS_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.1851257, 0.6783313, -0.2352601, 0.7538407, -0.9389664, 0.9135914
1: -0.3969364, 0.8526834, -0.4706453, 0.9488738, -1.3458102, 1.3233287
2: -0.3242711, 0.9644121, -0.3915833, 1.0723588, -1.3966299, 1.3559954
3: -0.7251614, 0.9704387, -0.8320177, 1.0878556, -1.8130170, 1.8024564
4: -0.6038694, 1.1718525, -0.7029035, 1.3031529, -1.9070222, 1.8747560

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0549649, upper bound: 1.0511174
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0550041, upper bound: 1.0508972
time: 0.40 seconds

## BFS IS instance: IS_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.2674521, 0.8277271, -0.2270867, 0.7398090, -1.0072610, 1.0548139
1: -0.5075184, 1.0548140, -0.4577317, 0.9312586, -1.4387770, 1.5125457
2: -0.4327361, 1.2001319, -0.3798388, 1.0540355, -1.4867716, 1.5799707
3: -0.9017408, 1.1985904, -0.8175005, 1.0655459, -1.9672867, 2.0160909
4: -0.7864016, 1.4086894, -0.6857330, 1.2797627, -2.0661645, 2.0944223

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_A2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505489, upper bound: 1.0503169
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502603, upper bound: 1.0501089
time: 0.41 seconds

## BFS IS instance: IS_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.2122585, 0.7096910, -0.2328624, 0.7488655, -0.9611241, 0.9425534
1: -0.4381096, 0.8818882, -0.4671082, 0.9423364, -1.3804460, 1.3489964
2: -0.3631817, 1.0126708, -0.3885702, 1.0657597, -1.4289414, 1.4012409
3: -0.7779223, 1.0290604, -0.8270060, 1.0809959, -1.8589182, 1.8560663
4: -0.6656082, 1.2431200, -0.6987019, 1.2962005, -1.9618087, 1.9418218

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_A1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0519984, upper bound: 1.0510306
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0515803, upper bound: 1.0508158
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.3022564, 0.8574660, -0.2246811, 0.7351621, -1.0374185, 1.0821471
1: -0.5588838, 1.0796481, -0.4541961, 0.9251198, -1.4840035, 1.5338442
2: -0.4826456, 1.2362378, -0.3768141, 1.0476735, -1.5303190, 1.6130519
3: -0.9591278, 1.2640674, -0.8124983, 1.0590022, -2.0181301, 2.0765657
4: -0.8620855, 1.4785453, -0.6815262, 1.2729509, -2.1350365, 2.1600714

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505489, upper bound: 1.0506176
time: 0.45 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502603, upper bound: 1.0504029
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.1851257, 0.6783313, -1.3271873, 2.7002649, -2.8853905, 2.0055187
1: -0.3969364, 0.8526834, -1.8266034, 2.9257355, -3.3226719, 2.6792867
2: -0.3242711, 0.9644121, -1.7866864, 3.3100519, -3.6343226, 2.7510986
3: -0.7251614, 0.9704387, -2.3538351, 3.8103127, -4.5354738, 3.3242738
4: -0.6038694, 1.1718525, -2.7129741, 3.8588223, -4.4626908, 3.8848267

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0488004, upper bound: 1.0401335
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_A1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0544207, upper bound: 1.0504677
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0544435, upper bound: 1.0491670
time: 0.45 seconds

## BFS IS instance: IS_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.2122585, 0.7096910, -1.3230309, 2.6886380, -2.9008965, 2.0327220
1: -0.4381096, 0.8818882, -1.8206375, 2.9125025, -3.3506122, 2.7025256
2: -0.3631817, 1.0126708, -1.7817578, 3.2965152, -3.6596963, 2.7944286
3: -0.7779223, 1.0290604, -2.3453972, 3.7946963, -4.5726185, 3.3744576
4: -0.6656082, 1.2431200, -2.7058206, 3.8440361, -4.5096445, 3.9489405

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_A2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0519357, upper bound: 1.0503808
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512565, upper bound: 1.0490802
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.2699254, 0.8529468, -1.3076617, 2.6589031, -2.9288285, 2.1606083
1: -0.5107998, 1.0817231, -1.8009224, 2.8803883, -3.3911877, 2.8826449
2: -0.4349182, 1.2233939, -1.7620664, 3.2624674, -3.6973855, 2.9854596
3: -0.9060124, 1.2251596, -2.3227785, 3.7524827, -4.6584949, 3.5479381
4: -0.7885253, 1.4212955, -2.6778111, 3.8049459, -4.5934715, 4.0991068

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_A1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504533, upper bound: 1.0492938
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499365, upper bound: 1.0483732
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.3204554, 0.9300180, -1.3036077, 2.6476357, -2.9680910, 2.2336257
1: -0.5845242, 1.1713222, -1.7951293, 2.8675218, -3.4520459, 2.9664516
2: -0.5056426, 1.3326001, -1.7572811, 3.2492998, -3.7549424, 3.0898809
3: -1.0023541, 1.3492649, -2.3145387, 3.7373593, -4.7397137, 3.6638033
4: -0.8973903, 1.5542920, -2.6708691, 3.7905874, -4.6879778, 4.2251611

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504533, upper bound: 1.0499679
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2

### Relational analysis result of IS_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500790, upper bound: 1.0486672
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -1.3271873, 2.7002649, -0.1851257, 0.6783313, -2.0055187, 2.8853903
1: -1.8266034, 2.9257355, -0.3969364, 0.8526834, -2.6792867, 3.3226719
2: -1.7866864, 3.3100519, -0.3242711, 0.9644121, -2.7510986, 3.6343231
3: -2.3538351, 3.8103127, -0.7251614, 0.9704387, -3.3242738, 4.5354738
4: -2.7129741, 3.8588223, -0.6038694, 1.1718525, -3.8848267, 4.4626908

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0401335, upper bound: 1.0488004
time: 0.39 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504677, upper bound: 1.0544207
time: 0.41 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0491670, upper bound: 1.0544435
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -1.3230309, 2.6886380, -0.2122585, 0.7096910, -2.0327220, 2.9008965
1: -1.8206375, 2.9125025, -0.4381096, 0.8818882, -2.7025256, 3.3506119
2: -1.7817578, 3.2965152, -0.3631817, 1.0126708, -2.7944286, 3.6596966
3: -2.3453972, 3.7946963, -0.7779223, 1.0290604, -3.3744576, 4.5726185
4: -2.7058206, 3.8440361, -0.6656082, 1.2431200, -3.9489405, 4.5096445

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B1_B2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0503808, upper bound: 1.0519357
time: 0.34 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0490802, upper bound: 1.0512565
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -1.3076617, 2.6589031, -0.2699254, 0.8529468, -2.1606083, 2.9288285
1: -1.8009224, 2.8803883, -0.5107998, 1.0817231, -2.8826444, 3.3911877
2: -1.7620664, 3.2624674, -0.4349182, 1.2233939, -2.9854598, 3.6973855
3: -2.3227785, 3.7524827, -0.9060124, 1.2251596, -3.5479381, 4.6584949
4: -2.6778111, 3.8049459, -0.7885253, 1.4212955, -4.0991068, 4.5934715

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B2_B1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0492938, upper bound: 1.0504533
time: 0.39 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0483732, upper bound: 1.0499365
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -1.3036077, 2.6476357, -0.3204554, 0.9300180, -2.2336257, 2.9680910
1: -1.7951293, 2.8675218, -0.5845242, 1.1713222, -2.9664516, 3.4520457
2: -1.7572811, 3.2492998, -0.5056426, 1.3326001, -3.0898809, 3.7549424
3: -2.3145387, 3.7373593, -1.0023541, 1.3492649, -3.6638036, 4.7397137
4: -2.6708691, 3.7905874, -0.8973903, 1.5542920, -4.2251611, 4.6879778

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B2_B2_A1

### Relational analysis result of IS_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499679, upper bound: 1.0507073
time: 0.41 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2

### Relational analysis result of IS_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0483732, upper bound: 1.0500790
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -1.3271873, 2.7002649, -1.2857423, 2.6045809, -3.9317682, 3.9860072
1: -1.8266034, 2.9257355, -1.7688465, 2.8143244, -4.6409273, 4.6945815
2: -1.7866864, 3.3100519, -1.7371318, 3.1887088, -4.9753952, 5.0471840
3: -2.3538351, 3.8103127, -2.2708325, 3.6756892, -6.0295238, 6.0811453
4: -2.7129741, 3.8588223, -2.6399686, 3.7237425, -6.4367166, 6.4987898

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_B1_B1_A1

### Relational analysis result of IS_A2_B2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474133, upper bound: 1.0480198
time: 0.37 seconds

## Relational analysis of IS_A2_B2_B1_B1_A2

### Relational analysis result of IS_A2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0474133, upper bound: 1.0496032
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -1.3230309, 2.6886380, -1.3498325, 2.6225345, -3.9455655, 4.0384703
1: -1.8206375, 2.9125025, -1.8129959, 2.8317785, -4.6524162, 4.7254982
2: -1.7817578, 3.2965152, -1.7776842, 3.2286239, -5.0103817, 5.0741997
3: -2.3453972, 3.7946963, -2.3294129, 3.7206798, -6.0660768, 6.1241093
4: -2.7058206, 3.8440361, -2.6985631, 3.7726209, -6.4784408, 6.5425987

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_B1_B2_A1

### Relational analysis result of IS_A2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0495975, upper bound: 1.0490352
time: 0.41 seconds

## Relational analysis of IS_A2_B2_B1_B2_A2

### Relational analysis result of IS_A2_B2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0482969, upper bound: 1.0483561
time: 0.37 seconds

## BFS IS instance: IS_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -1.3036077, 2.6476357, -1.3900077, 2.7811232, -4.0847311, 4.0376434
1: -1.7951293, 2.8675218, -1.8534610, 3.0589395, -4.8540678, 4.7209821
2: -1.7572811, 3.2492998, -1.7888100, 3.4775665, -5.2348475, 5.0381098
3: -2.3145387, 3.7373593, -2.4571981, 3.9132597, -6.2277980, 6.1945572
4: -2.6708691, 3.7905874, -2.7346168, 3.9865932, -6.6574621, 6.5252037

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0496440, upper bound: 1.0490225
time: 0.38 seconds

## Relational analysis of IS_A2_B2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0483434, upper bound: 1.0483434
time: 0.36 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.44 seconds
IS_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0549649, upper bound: 1.0511174
IS_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0550041, upper bound: 1.0508972
IS_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0505489, upper bound: 1.0503169
IS_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0502603, upper bound: 1.0501089
IS_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0519984, upper bound: 1.0510306
IS_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0515803, upper bound: 1.0508158
IS_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0505489, upper bound: 1.0506176
IS_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0502603, upper bound: 1.0504029
IS_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0544207, upper bound: 1.0504677
IS_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0544435, upper bound: 1.0491670
IS_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0519357, upper bound: 1.0503808
IS_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0512565, upper bound: 1.0490802
IS_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0504533, upper bound: 1.0492938
IS_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0499365, upper bound: 1.0483732
IS_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0504533, upper bound: 1.0499679
IS_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0500790, upper bound: 1.0486672
IS_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0504677, upper bound: 1.0544207
IS_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0491670, upper bound: 1.0544435
IS_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0503808, upper bound: 1.0519357
IS_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0490802, upper bound: 1.0512565
IS_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0492938, upper bound: 1.0504533
IS_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0483732, upper bound: 1.0499365
IS_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0499679, upper bound: 1.0507073
IS_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0483732, upper bound: 1.0500790
IS_A2_B2_B1_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0474133, upper bound: 1.0480198
IS_A2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0474133, upper bound: 1.0496032
IS_A2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0495975, upper bound: 1.0490352
IS_A2_B2_B1_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0482969, upper bound: 1.0483561
IS_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0496440, upper bound: 1.0490225
IS_A2_B2_B2_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.44
Output dim: 0, lower bound: -1.0483434, upper bound: 1.0483434

## BFS IS instance: IS_A1_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1835381, 0.6760455, -0.1999423, 0.7006042, -0.8841423, 0.8759879
1: -0.3945316, 0.8501992, -0.4179226, 0.8853608, -1.2798924, 1.2681217
2: -0.3219616, 0.9609872, -0.3408493, 0.9954937, -1.3174553, 1.3018365
3: -0.7221040, 0.9668397, -0.7652031, 1.0039408, -1.7260448, 1.7320428
4: -0.6003585, 1.1673317, -0.6261964, 1.2012441, -1.8016026, 1.7935281

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0545562, upper bound: 1.0511174
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0545562, upper bound: 1.0511174
time: 0.40 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1851257, 0.6783313, -0.2121059, 0.7206463, -0.9057720, 0.8904372
1: -0.3969364, 0.8526834, -0.4365718, 0.9153755, -1.3123119, 1.2892551
2: -0.3242711, 0.9644121, -0.3573927, 1.0247047, -1.3489758, 1.3218048
3: -0.7251614, 0.9704387, -0.7891545, 1.0367652, -1.7619267, 1.7595932
4: -0.6038694, 1.1718525, -0.6509602, 1.2383894, -1.8422588, 1.8228127

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0547673, upper bound: 1.0507519
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0547673, upper bound: 1.0508972
time: 0.38 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.2659495, 0.8254572, -0.1925148, 0.6889513, -0.9549008, 1.0179720
1: -0.5054118, 1.0523742, -0.4058403, 0.8711594, -1.3765712, 1.4582145
2: -0.4305603, 1.1968483, -0.3296701, 0.9800427, -1.4106030, 1.5265183
3: -0.8989345, 1.1950972, -0.7520148, 0.9850789, -1.8840134, 1.9471121
4: -0.7830958, 1.4043270, -0.6096900, 1.1804553, -1.9635510, 2.0140171

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504589, upper bound: 1.0501382
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505489, upper bound: 1.0503169
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505489, upper bound: 1.0503169
time: 0.35 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.2674521, 0.8277271, -0.2037833, 0.7069973, -0.9744495, 1.0315104
1: -0.5075184, 1.0548140, -0.4233948, 0.8982641, -1.4057825, 1.4782088
2: -0.4327361, 1.2001319, -0.3453550, 1.0066265, -1.4393625, 1.5454869
3: -0.9017408, 1.1985904, -0.7744057, 1.0150458, -1.9167866, 1.9729960
4: -0.7864016, 1.4086894, -0.6332868, 1.2156501, -2.0020518, 2.0419762

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499663, upper bound: 1.0499663
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499663, upper bound: 1.0501089
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.2107418, 0.7074690, -0.1983184, 0.6975295, -0.9082713, 0.9057873
1: -0.4358593, 0.8795447, -0.4154836, 0.8815370, -1.3173964, 1.2950283
2: -0.3609553, 1.0093845, -0.3386825, 0.9909576, -1.3519129, 1.3480670
3: -0.7750789, 1.0255320, -0.7617351, 0.9996639, -1.7747428, 1.7872672
4: -0.6622281, 1.2387280, -0.6230548, 1.1963723, -1.8586004, 1.8617828

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0515803, upper bound: 1.0506128
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0515803, upper bound: 1.0506128
time: 0.37 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2122585, 0.7096910, -0.2096955, 0.7159536, -0.9282122, 0.9193865
1: -0.4381096, 0.8818882, -0.4329281, 0.9091793, -1.3472890, 1.3148162
2: -0.3631817, 1.0126708, -0.3543391, 1.0183605, -1.3815422, 1.3670099
3: -0.7779223, 1.0290604, -0.7839895, 1.0300579, -1.8079803, 1.8130499
4: -0.6656082, 1.2431200, -0.6466651, 1.2317281, -1.8973362, 1.8897851

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_A1_B2_B1

### Relational analysis result of IS_A1_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512864, upper bound: 1.0506733
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512864, upper bound: 1.0507308
time: 0.40 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.3007503, 0.8551875, -0.1910050, 0.6859025, -0.9866528, 1.0461925
1: -0.5567948, 1.0772043, -0.4033984, 0.8673967, -1.4241915, 1.4806027
2: -0.4804639, 1.2327573, -0.3274869, 0.9755512, -1.4560151, 1.5602442
3: -0.9563015, 1.2604315, -0.7485576, 0.9808752, -1.9371767, 2.0089891
4: -0.8587720, 1.4739208, -0.6065688, 1.1756215, -2.0343935, 2.0804896

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501999, upper bound: 1.0501999
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501999, upper bound: 1.0501999
time: 0.42 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.3022564, 0.8574660, -0.2013619, 0.7026125, -1.0048690, 1.0588279
1: -0.5588838, 1.0796481, -0.4197503, 0.8923464, -1.4512303, 1.4993985
2: -0.4826456, 1.2362378, -0.3422847, 1.0006269, -1.4832726, 1.5785224
3: -0.9591278, 1.2640674, -0.7692537, 1.0085855, -1.9677134, 2.0333211
4: -0.8620855, 1.4785453, -0.6289783, 1.2092023, -2.0712876, 2.1075237

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501089, upper bound: 1.0502603
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501089, upper bound: 1.0504029
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1835381, 0.6760455, -1.2651416, 2.5885944, -2.7721324, 1.9411871
1: -0.3945316, 0.8501992, -1.7519391, 2.8037324, -3.1982639, 2.6021383
2: -0.3219616, 0.9609872, -1.7190838, 3.1656418, -3.4876034, 2.6800709
3: -0.7221040, 0.9668397, -2.2556522, 3.6565268, -4.3786306, 3.2224917
4: -0.6003585, 1.1673317, -2.6154146, 3.6987345, -4.2990932, 3.7827463

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0485091, upper bound: 1.0374493
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0543227, upper bound: 1.0504677
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0543227, upper bound: 1.0504677
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1851257, 0.6783313, -1.2914166, 2.6484618, -2.8335872, 1.9697480
1: -0.3969364, 0.8526834, -1.7824845, 2.8723085, -3.2692444, 2.6351678
2: -0.3242711, 0.9644121, -1.7462265, 3.2360384, -3.5603094, 2.7106385
3: -0.7251614, 0.9704387, -2.2972631, 3.7364984, -4.4616594, 3.2677019
4: -0.6038694, 1.1718525, -2.6558306, 3.7713811, -4.3752503, 3.8276830

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0485091, upper bound: 1.0393975
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0540514, upper bound: 1.0491670
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0544435, upper bound: 1.0491670
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.2107418, 0.7074690, -1.2609853, 2.5771461, -2.7878876, 1.9684542
1: -0.4358593, 0.8795447, -1.7460914, 2.7907536, -3.2266128, 2.6256361
2: -0.3609553, 1.0093845, -1.7142205, 3.1523950, -3.5133500, 2.7236052
3: -0.7750789, 1.0255320, -2.2473607, 3.6410317, -4.4161105, 3.2728927
4: -0.6622281, 1.2387280, -2.6084113, 3.6842589, -4.3464870, 3.8471394

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512565, upper bound: 1.0490802
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512565, upper bound: 1.0490802
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.2122585, 0.7096910, -1.2870837, 2.6366985, -2.8489571, 1.9967747
1: -0.4381096, 0.8818882, -1.7764096, 2.8588827, -3.2969921, 2.6582978
2: -0.3631817, 1.0126708, -1.7412047, 3.2223477, -3.5855289, 2.7538755
3: -0.7779223, 1.0290604, -2.2886417, 3.7204709, -4.4983931, 3.3177021
4: -0.6656082, 1.2431200, -2.6484923, 3.7564254, -4.4220333, 3.8916123

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_A2_B2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0503540, upper bound: 1.0486297
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0503540, upper bound: 1.0486761
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.2683378, 0.8503048, -1.2449245, 2.5461495, -2.8144870, 2.0952294
1: -0.5085518, 1.0787544, -1.7254748, 2.7570257, -3.2655776, 2.8042293
2: -0.4326545, 1.2197943, -1.6936946, 3.1166339, -3.5492883, 2.9134881
3: -0.9030173, 1.2211714, -2.2236793, 3.5969324, -4.4999495, 3.4448507
4: -0.7851263, 1.4167428, -2.5791800, 3.6432774, -4.4284039, 3.9959223

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0503338, upper bound: 1.0488995
time: 0.45 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504370, upper bound: 1.0492938
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504370, upper bound: 1.0492938
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2699254, 0.8529468, -1.2719434, 2.6069660, -2.8768914, 2.1248901
1: -0.5107998, 1.0817231, -1.7567315, 2.8267782, -3.3375776, 2.8384542
2: -0.4349182, 1.2233939, -1.7214665, 3.1882508, -3.6231689, 2.9448605
3: -0.9060124, 1.2251596, -2.2661893, 3.6783309, -4.5843430, 3.4913490
4: -0.7885253, 1.4212955, -2.6204011, 3.7172136, -4.5057392, 4.0416965

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0498917, upper bound: 1.0482416
time: 0.32 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0444293, upper bound: 1.0431251
time: 0.45 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.3188048, 0.9272805, -1.2408764, 2.5350981, -2.8539028, 2.1681566
1: -0.5822500, 1.1682063, -1.7198091, 2.7445037, -3.3267536, 2.8880153
2: -0.5033102, 1.3287069, -1.6889806, 3.1038139, -3.6071241, 3.0176873
3: -0.9992414, 1.3450904, -2.2156267, 3.5820315, -4.5812726, 3.5607171
4: -0.8938787, 1.5493380, -2.5723977, 3.6292882, -4.5231667, 4.1217356

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0498760, upper bound: 1.0486672
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0498760, upper bound: 1.0486672
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.3204554, 0.9300180, -1.2677133, 2.5955524, -2.9160078, 2.1977313
1: -0.5845242, 1.1713222, -1.7508230, 2.8137093, -3.3982334, 2.9221454
2: -0.5056426, 1.3326001, -1.7165847, 3.1749177, -3.6805604, 3.0491848
3: -1.0023541, 1.3492649, -2.2577801, 3.6627932, -4.6651473, 3.6070447
4: -0.8973903, 1.5542920, -2.6132686, 3.7026765, -4.6000667, 4.1675606

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0491766, upper bound: 1.0482167
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_B2

### Relational analysis result of IS_A1_B2_A2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0491766, upper bound: 1.0486524
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -1.2651416, 2.5885944, -0.1835381, 0.6760455, -1.9411871, 2.7721324
1: -1.7519391, 2.8037324, -0.3945316, 0.8501992, -2.6021383, 3.1982641
2: -1.7190838, 3.1656418, -0.3219616, 0.9609872, -2.6800709, 3.4876034
3: -2.2556522, 3.6565268, -0.7221040, 0.9668397, -3.2224917, 4.3786306
4: -2.6154146, 3.6987345, -0.6003585, 1.1673317, -3.7827463, 4.2990932

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0374493, upper bound: 1.0485091
time: 0.36 seconds

## Relational analysis of IS_A2_B1_B1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B1_B1_A1_B1

### Relational analysis result of IS_A2_B1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504677, upper bound: 1.0543227
time: 0.38 seconds

## Relational analysis of IS_A2_B1_B1_B1_A1_B2

### Relational analysis result of IS_A2_B1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504677, upper bound: 1.0544207
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -1.2914166, 2.6484618, -0.1851257, 0.6783313, -1.9697480, 2.8335872
1: -1.7824845, 2.8723085, -0.3969364, 0.8526834, -2.6351678, 3.2692449
2: -1.7462265, 3.2360384, -0.3242711, 0.9644121, -2.7106385, 3.5603094
3: -2.2972631, 3.7364984, -0.7251614, 0.9704387, -3.2677019, 4.4616594
4: -2.6558306, 3.7713811, -0.6038694, 1.1718525, -3.8276830, 4.3752503

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0393975, upper bound: 1.0486733
time: 0.38 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0491670, upper bound: 1.0540514
time: 0.35 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_B2

### Relational analysis result of IS_A2_B1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0491670, upper bound: 1.0544435
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -1.2609853, 2.5771461, -0.2107418, 0.7074690, -1.9684542, 2.7878876
1: -1.7460914, 2.7907536, -0.4358593, 0.8795447, -2.6256361, 3.2266128
2: -1.7142205, 3.1523950, -0.3609553, 1.0093845, -2.7236052, 3.5133500
3: -2.2473607, 3.6410317, -0.7750789, 1.0255320, -3.2728927, 4.4161100
4: -2.6084113, 3.6842589, -0.6622281, 1.2387280, -3.8471394, 4.3464870

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0490802, upper bound: 1.0512565
time: 0.38 seconds

## Relational analysis of IS_A2_B1_B1_B2_A1_B2

### Relational analysis result of IS_A2_B1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0490802, upper bound: 1.0512565
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -1.2870837, 2.6366985, -0.2122585, 0.7096910, -1.9967747, 2.8489571
1: -1.7764096, 2.8588827, -0.4381096, 0.8818882, -2.6582978, 3.2969918
2: -1.7412047, 3.2223477, -0.3631817, 1.0126708, -2.7538755, 3.5855289
3: -2.2886417, 3.7204709, -0.7779223, 1.0290604, -3.3177021, 4.4983931
4: -2.6484923, 3.7564254, -0.6656082, 1.2431200, -3.8916123, 4.4220333

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_B1_B2_A2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0486297, upper bound: 1.0503540
time: 0.43 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0486297, upper bound: 1.0512565
time: 0.44 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -1.2449245, 2.5461495, -0.2683378, 0.8503048, -2.0952294, 2.8144870
1: -1.7254748, 2.7570257, -0.5085518, 1.0787544, -2.8042293, 3.2655776
2: -1.6936946, 3.1166339, -0.4326545, 1.2197943, -2.9134881, 3.5492883
3: -2.2236793, 3.5969324, -0.9030173, 1.2211714, -3.4448507, 4.4999485
4: -2.5791800, 3.6432774, -0.7851263, 1.4167428, -3.9959221, 4.4284039

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B2_B1_A1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0488995, upper bound: 1.0503338
time: 0.37 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_B2_B1_A1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0492938, upper bound: 1.0504370
time: 0.38 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0492938, upper bound: 1.0504533
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -1.2719434, 2.6069660, -0.2699254, 0.8529468, -2.1248901, 2.8768914
1: -1.7567315, 2.8267782, -0.5107998, 1.0817231, -2.8384540, 3.3375778
2: -1.7214665, 3.1882508, -0.4349182, 1.2233939, -2.9448605, 3.6231689
3: -2.2661893, 3.6783309, -0.9060124, 1.2251596, -3.4913490, 4.5843430
4: -2.6204011, 3.7172136, -0.7885253, 1.4212955, -4.0416965, 4.5057392

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B2_B1_A2_A1

### Relational analysis result of IS_A2_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0482416, upper bound: 1.0498917
time: 0.38 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2_A2

### Relational analysis result of IS_A2_B1_B2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0431251, upper bound: 1.0444293
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -1.2408764, 2.5350981, -0.3188048, 0.9272805, -2.1681566, 2.8539028
1: -1.7198091, 2.7445037, -0.5822500, 1.1682063, -2.8880153, 3.3267534
2: -1.6889806, 3.1038139, -0.5033102, 1.3287069, -3.0176871, 3.6071241
3: -2.2156267, 3.5820315, -0.9992414, 1.3450904, -3.5607171, 4.5812731
4: -2.5723977, 3.6292882, -0.8938787, 1.5493380, -4.1217356, 4.5231667

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B2_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0486672, upper bound: 1.0498760
time: 0.41 seconds

## Relational analysis of IS_A2_B1_B2_B2_A1_B2

### Relational analysis result of IS_A2_B1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0486672, upper bound: 1.0500790
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -1.2677133, 2.5955524, -0.3204554, 0.9300180, -2.1977313, 2.9160078
1: -1.7508230, 2.8137093, -0.5845242, 1.1713222, -2.9221454, 3.3982334
2: -1.7165847, 3.1749177, -0.5056426, 1.3326001, -3.0491848, 3.6805604
3: -2.2577801, 3.6627932, -1.0023541, 1.3492649, -3.6070449, 4.6651468
4: -2.6132686, 3.7026765, -0.8973903, 1.5542920, -4.1675601, 4.6000667

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_B2_B2_A2_A1

### Relational analysis result of IS_A2_B1_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479227, upper bound: 1.0491766
time: 0.38 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B2_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0435146, upper bound: 1.0446084
time: 0.38 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B2_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0486672, upper bound: 1.0498760
time: 0.39 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0486672, upper bound: 1.0500790
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -1.3654406, 2.6592798, -1.2857423, 2.6045809, -3.9700212, 3.9450221
1: -1.8352041, 2.8745258, -1.7688465, 2.8143244, -4.6495285, 4.6433725
2: -1.7965705, 3.2761538, -1.7371318, 3.1887088, -4.9852786, 5.0132856
3: -2.3616958, 3.7730355, -2.2708325, 3.6756892, -6.0373845, 6.0438681
4: -2.7261250, 3.8257360, -2.6399686, 3.7237425, -6.4498672, 6.4657049

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_B1_B1_A2_A1

### Relational analysis result of IS_A2_B2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0474133, upper bound: 1.0496032
time: 0.39 seconds

## Relational analysis of IS_A2_B2_B1_B1_A2_A2

### Relational analysis result of IS_A2_B2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0474133, upper bound: 1.0496032
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -1.2609853, 2.5771461, -1.3456051, 2.6162264, -3.8772116, 3.9227512
1: -1.7460914, 2.7907536, -1.8085377, 2.8248672, -4.5709581, 4.5992913
2: -1.7142205, 3.1523950, -1.7737784, 3.2205167, -4.9347367, 4.9261727
3: -2.2473607, 3.6410317, -2.3232698, 3.7117167, -5.9590774, 5.9643006
4: -2.6084113, 3.6842589, -2.6929026, 3.7635832, -6.3719945, 6.3771615

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_B1_B2_A1_A1

### Relational analysis result of IS_A2_B2_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0304890, upper bound: 1.0320348
time: 0.46 seconds

## Relational analysis of IS_A2_B2_B1_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_B1_B2_A1_A1

### Relational analysis result of IS_A2_B2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0495975, upper bound: 1.0490352
time: 0.41 seconds

## Relational analysis of IS_A2_B2_B1_B2_A1_A2

### Relational analysis result of IS_A2_B2_B1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0488641, upper bound: 1.0490352
time: 0.39 seconds

## BFS IS instance: IS_A2_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -1.2408764, 2.5350981, -1.3857040, 2.7748103, -4.0156865, 3.9208019
1: -1.7198091, 2.7445037, -1.8483119, 3.0518951, -4.7717042, 4.5928154
2: -1.6889806, 3.1038139, -1.7843323, 3.4695272, -5.1585064, 4.8881454
3: -2.2156267, 3.5820315, -2.4509192, 3.9041967, -6.1198235, 6.0329499
4: -2.5723977, 3.6292882, -2.7282419, 3.9775968, -6.5499945, 6.3575296

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_B2_B2_A1_A1

### Relational analysis result of IS_A2_B2_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0419084, upper bound: 1.0429551
time: 0.42 seconds

## Relational analysis of IS_A2_B2_B2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_B2_B2_A1_B1

### Relational analysis result of IS_A2_B2_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0483434, upper bound: 1.0483434
time: 0.48 seconds

## Relational analysis of IS_A2_B2_B2_B2_A1_B2

### Relational analysis result of IS_A2_B2_B2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0483434, upper bound: 1.0483434
time: 0.45 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 7.07 seconds
IS_A1_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0545562, upper bound: 1.0511174
IS_A1_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0545562, upper bound: 1.0511174
IS_A1_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0547673, upper bound: 1.0507519
IS_A1_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0547673, upper bound: 1.0508972
IS_A1_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0505489, upper bound: 1.0503169
IS_A1_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0505489, upper bound: 1.0503169
IS_A1_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0499663, upper bound: 1.0499663
IS_A1_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0499663, upper bound: 1.0501089
IS_A1_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0515803, upper bound: 1.0506128
IS_A1_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0515803, upper bound: 1.0506128
IS_A1_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0512864, upper bound: 1.0506733
IS_A1_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0512864, upper bound: 1.0507308
IS_A1_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0501999, upper bound: 1.0501999
IS_A1_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0501999, upper bound: 1.0501999
IS_A1_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0501089, upper bound: 1.0502603
IS_A1_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0501089, upper bound: 1.0504029
IS_A1_B2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0543227, upper bound: 1.0504677
IS_A1_B2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0543227, upper bound: 1.0504677
IS_A1_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0540514, upper bound: 1.0491670
IS_A1_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0544435, upper bound: 1.0491670
IS_A1_B2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0512565, upper bound: 1.0490802
IS_A1_B2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0512565, upper bound: 1.0490802
IS_A1_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0503540, upper bound: 1.0486297
IS_A1_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0503540, upper bound: 1.0486761
IS_A1_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0504370, upper bound: 1.0492938
IS_A1_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0504370, upper bound: 1.0492938
IS_A1_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0498917, upper bound: 1.0482416
IS_A1_B2_A2_A1_B2_B2, status: Status.VERIFIED, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0444293, upper bound: 1.0431251
IS_A1_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0498760, upper bound: 1.0486672
IS_A1_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0498760, upper bound: 1.0486672
IS_A1_B2_A2_A2_B2_B1, status: Status.VERIFIED, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0491766, upper bound: 1.0482167
IS_A1_B2_A2_A2_B2_B2, status: Status.VERIFIED, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0491766, upper bound: 1.0486524
IS_A2_B1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0504677, upper bound: 1.0543227
IS_A2_B1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0504677, upper bound: 1.0544207
IS_A2_B1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0491670, upper bound: 1.0540514
IS_A2_B1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0491670, upper bound: 1.0544435
IS_A2_B1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0490802, upper bound: 1.0512565
IS_A2_B1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0490802, upper bound: 1.0512565
IS_A2_B1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0486297, upper bound: 1.0503540
IS_A2_B1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0486297, upper bound: 1.0512565
IS_A2_B1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0492938, upper bound: 1.0504370
IS_A2_B1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0492938, upper bound: 1.0504533
IS_A2_B1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0482416, upper bound: 1.0498917
IS_A2_B1_B2_B1_A2_A2, status: Status.VERIFIED, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0431251, upper bound: 1.0444293
IS_A2_B1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0486672, upper bound: 1.0498760
IS_A2_B1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0486672, upper bound: 1.0500790
IS_A2_B1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0486672, upper bound: 1.0498760
IS_A2_B1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0486672, upper bound: 1.0500790
IS_A2_B2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0474133, upper bound: 1.0496032
IS_A2_B2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0474133, upper bound: 1.0496032
IS_A2_B2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0495975, upper bound: 1.0490352
IS_A2_B2_B1_B2_A1_A2, status: Status.VERIFIED, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0488641, upper bound: 1.0490352
IS_A2_B2_B2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0483434, upper bound: 1.0483434
IS_A2_B2_B2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 7.07
Output dim: 0, lower bound: -1.0483434, upper bound: 1.0483434

## BFS IS instance: IS_A1_B1_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.2082714, 0.7079026, -0.1998790, 0.7004954, -0.9087669, 0.9077816
1: -0.4245980, 0.8958343, -0.4178324, 0.8852251, -1.3098230, 1.3136667
2: -0.3538412, 1.0096250, -0.3407625, 0.9953208, -1.3491620, 1.3503875
3: -0.7711517, 1.0166786, -0.7650728, 1.0037713, -1.7749230, 1.7817514
4: -0.6532400, 1.2142799, -0.6260678, 1.2010533, -1.8542933, 1.8403476

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0545562, upper bound: 1.0510595
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0528772, upper bound: 1.0511174
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_A1_A2

### Relational analysis result of IS_A1_B1_A1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0528772, upper bound: 1.0511174
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1730760, 0.6638508, -0.1999423, 0.7006042, -0.8736802, 0.8637931
1: -0.3801277, 0.8354926, -0.4179226, 0.8853608, -1.2654885, 1.2534151
2: -0.3072035, 0.9425610, -0.3408493, 0.9954937, -1.3026972, 1.2834103
3: -0.7048930, 0.9456613, -0.7652031, 1.0039408, -1.7088338, 1.7108643
4: -0.5783734, 1.1440246, -0.6261964, 1.2012441, -1.7796175, 1.7702210

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0491555, upper bound: 1.0399978
time: 0.45 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0549649, upper bound: 1.0510559
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0528772, upper bound: 1.0511174
time: 0.48 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0528772, upper bound: 1.0511174
time: 0.37 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.1851257, 0.6783313, -0.1796174, 0.6729426, -0.8580683, 0.8579487
1: -0.3969364, 0.8526834, -0.3886608, 0.8546216, -1.2515581, 1.2413442
2: -0.3242711, 0.9644121, -0.3137678, 0.9577398, -1.2820108, 1.2781799
3: -0.7251614, 0.9704387, -0.7210382, 0.9618257, -1.6869872, 1.6914769
4: -0.6038694, 1.1718525, -0.5868647, 1.1561475, -1.7600169, 1.7587172

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A1_B2_B1_A1

### Relational analysis result of IS_A1_B1_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0534637, upper bound: 1.0505998
time: 0.48 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_B1_A2

### Relational analysis result of IS_A1_B1_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0547673, upper bound: 1.0507519
time: 0.41 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.1851257, 0.6783313, -0.2148956, 0.7115479, -0.8966736, 0.8932270
1: -0.3969364, 0.8526834, -0.4401824, 0.8935100, -1.2904465, 1.2928658
2: -0.3242711, 0.9644121, -0.3639436, 1.0145798, -1.3388509, 1.3283557
3: -0.7251614, 0.9704387, -0.7847203, 1.0327761, -1.7579376, 1.7551590
4: -0.6038694, 1.1718525, -0.6645471, 1.2404692, -1.8443387, 1.8363996

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A1_B2_B2_A1

### Relational analysis result of IS_A1_B1_A1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0534637, upper bound: 1.0508131
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_B2_A2

### Relational analysis result of IS_A1_B1_A1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0547673, upper bound: 1.0508972
time: 0.45 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.2659495, 0.8254572, -0.1797266, 0.6730186, -0.9389681, 1.0051838
1: -0.5054118, 1.0523742, -0.3887473, 0.8507720, -1.3561838, 1.4411216
2: -0.4305603, 1.1968483, -0.3128836, 0.9551901, -1.3857504, 1.5097319
3: -0.8989345, 1.1950972, -0.7242810, 0.9599463, -1.8588808, 1.9193782
4: -0.7830958, 1.4043270, -0.5851926, 1.1522350, -1.9353309, 1.9895196

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504589, upper bound: 1.0501335
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4

Time for candidate selection: 4.94 seconds

### Candidate
type: A, layer: 3, pos: 31

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0465749, upper bound: 1.0473786
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_B2

### Relational analysis result of IS_A1_B1_A1_A2_B1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0467681, upper bound: 1.0473742
time: 0.40 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.2659495, 0.8254572, -0.2493654, 0.7847975, -1.0507470, 1.0748227
1: -0.5054118, 1.0523742, -0.4858735, 1.0028971, -1.5083089, 1.5382478
2: -0.4305603, 1.1968483, -0.4051812, 1.1277900, -1.5583503, 1.6020296
3: -0.8989345, 1.1950972, -0.8688342, 1.1415675, -2.0405021, 2.0639315
4: -0.7830958, 1.4043270, -0.7417756, 1.3354181, -2.1185138, 2.1461027

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504589, upper bound: 1.0501335
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 11

Time for candidate selection: 4.97 seconds

### Candidate
type: A, layer: 3, pos: 31

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0465749, upper bound: 1.0473805
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0467681, upper bound: 1.0473805
time: 0.38 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.2674521, 0.8277271, -0.1720077, 0.6606607, -0.9281129, 0.9997348
1: -0.5075184, 1.0548140, -0.3763350, 0.8395613, -1.3470798, 1.4311490
2: -0.4327361, 1.2001319, -0.3024756, 0.9408650, -1.3736012, 1.5026075
3: -0.9017408, 1.1985904, -0.7069268, 0.9420997, -1.8438405, 1.9055172
4: -0.7864016, 1.4086894, -0.5702088, 1.1342547, -1.9206563, 1.9788982

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_B1

### Relational analysis result of IS_A1_B1_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499216, upper bound: 1.0498500
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4

Time for candidate selection: 5.25 seconds

### Candidate
type: A, layer: 3, pos: 31

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_B1

### Relational analysis result of IS_A1_B1_A1_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0463104, upper bound: 1.0470466
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_B2

### Relational analysis result of IS_A1_B1_A1_A2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0439919, upper bound: 1.0439919
time: 0.41 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.2674521, 0.8277271, -0.2077064, 0.7004760, -0.9679281, 1.0354335
1: -0.5075184, 1.0548140, -0.4286225, 0.8807602, -1.3882786, 1.4834365
2: -0.4327361, 1.2001319, -0.3532239, 1.0001026, -1.4328387, 1.5533558
3: -0.9017408, 1.1985904, -0.7721944, 1.0146000, -1.9163408, 1.9707848
4: -0.7864016, 1.4086894, -0.6487206, 1.2206337, -2.0070353, 2.0574100

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499216, upper bound: 1.0500052
time: 0.45 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499663, upper bound: 1.0501089
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499663, upper bound: 1.0501089
time: 0.43 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1806883, 0.6643587, -0.1983184, 0.6975295, -0.8782178, 0.8626771
1: -0.3907865, 0.8332261, -0.4154836, 0.8815370, -1.2723236, 1.2487097
2: -0.3165183, 0.9454506, -0.3386825, 0.9909576, -1.3074758, 1.2841332
3: -0.7177029, 0.9569116, -0.7617351, 0.9996639, -1.7173668, 1.7186468
4: -0.5945891, 1.1518157, -0.6230548, 1.1963723, -1.7909614, 1.7748704

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0519984, upper bound: 1.0510306
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0519984, upper bound: 1.0510306
time: 0.38 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1905538, 0.6793638, -0.1983184, 0.6975295, -0.8880833, 0.8776822
1: -0.4057388, 0.8529248, -0.4154836, 0.8815370, -1.2872758, 1.2684083
2: -0.3302413, 0.9696531, -0.3386825, 0.9909576, -1.3211988, 1.3083357
3: -0.7378664, 0.9809170, -0.7617351, 0.9996639, -1.7375304, 1.7426522
4: -0.6153858, 1.1839089, -0.6230548, 1.1963723, -1.8117580, 1.8069637

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0519984, upper bound: 1.0510306
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0519984, upper bound: 1.0510306
time: 0.36 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.2122585, 0.7096910, -0.1796174, 0.6729426, -0.8852012, 0.8893083
1: -0.4381096, 0.8818882, -0.3886608, 0.8546216, -1.2927313, 1.2705489
2: -0.3631817, 1.0126708, -0.3137678, 0.9577398, -1.3209214, 1.3264385
3: -0.7779223, 1.0290604, -0.7210382, 0.9618257, -1.7397480, 1.7500986
4: -0.6656082, 1.2431200, -0.5868647, 1.1561475, -1.8217556, 1.8299847

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512864, upper bound: 1.0506733
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512864, upper bound: 1.0506733
time: 0.42 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.2122585, 0.7096910, -0.2148956, 0.7115479, -0.9238064, 0.9245867
1: -0.4381096, 0.8818882, -0.4401824, 0.8935100, -1.3316197, 1.3220706
2: -0.3631817, 1.0126708, -0.3639436, 1.0145798, -1.3777615, 1.3766143
3: -0.7779223, 1.0290604, -0.7847203, 1.0327761, -1.8106985, 1.8137807
4: -0.6656082, 1.2431200, -0.6645471, 1.2404692, -1.9060774, 1.9076672

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_B1

### Relational analysis result of IS_A1_B1_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512864, upper bound: 1.0507308
time: 0.47 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512864, upper bound: 1.0507308
time: 0.41 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.2689438, 0.8082717, -0.1910050, 0.6859025, -0.9548463, 0.9992768
1: -0.5119697, 1.0272677, -0.4033984, 0.8673967, -1.3793664, 1.4306661
2: -0.4340174, 1.1612520, -0.3274869, 0.9755512, -1.4095685, 1.4887389
3: -0.8960808, 1.1860839, -0.7485576, 0.9808752, -1.8769560, 1.9346415
4: -0.7881429, 1.3786812, -0.6065688, 1.1756215, -1.9637644, 1.9852500

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506179, upper bound: 1.0506176
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506179, upper bound: 1.0506176
time: 0.42 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.2798769, 0.8257407, -0.1910050, 0.6859025, -0.9657795, 1.0167457
1: -0.5248469, 1.0494590, -0.4033984, 0.8673967, -1.3922436, 1.4528574
2: -0.4479707, 1.1899524, -0.3274869, 0.9755512, -1.4235220, 1.5174392
3: -0.9173980, 1.2125832, -0.7485576, 0.9808752, -1.8982732, 1.9611408
4: -0.8094229, 1.4129685, -0.6065688, 1.1756215, -1.9850444, 2.0195372

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506179, upper bound: 1.0506176
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506179, upper bound: 1.0506176
time: 0.43 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.3022564, 0.8574660, -0.1720077, 0.6606607, -0.9629171, 1.0294737
1: -0.5588838, 1.0796481, -0.3763350, 0.8395613, -1.3984451, 1.4559832
2: -0.4826456, 1.2362378, -0.3024756, 0.9408650, -1.4235106, 1.5387133
3: -0.9591278, 1.2640674, -0.7069268, 0.9420997, -1.9012275, 1.9709942
4: -0.8620855, 1.4785453, -0.5702088, 1.1342547, -1.9963402, 2.0487542

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_A2_B2_B1_B1

### Relational analysis result of IS_A1_B1_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501089, upper bound: 1.0502603
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2_B1_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501089, upper bound: 1.0502603
time: 0.49 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.3022564, 0.8574660, -0.2077064, 0.7004760, -1.0027324, 1.0651724
1: -0.5588838, 1.0796481, -0.4286225, 0.8807602, -1.4396440, 1.5082706
2: -0.4826456, 1.2362378, -0.3532239, 1.0001026, -1.4827483, 1.5894617
3: -0.9591278, 1.2640674, -0.7721944, 1.0146000, -1.9737279, 2.0362618
4: -0.8620855, 1.4785453, -0.6487206, 1.2206337, -2.0827193, 2.1272659

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_A2_B2_B2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501089, upper bound: 1.0504029
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2_B2_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501089, upper bound: 1.0504029
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.2082714, 0.7079026, -1.2650044, 2.5882788, -2.7965503, 1.9729069
1: -0.4245980, 0.8958343, -1.7517481, 2.8033786, -3.2279766, 2.6475825
2: -0.3538412, 1.0096250, -1.7189114, 3.1652446, -3.5190854, 2.7285364
3: -0.7711517, 1.0166786, -2.2553968, 3.6560674, -4.4272180, 3.2720754
4: -0.6532400, 1.2142799, -2.6151581, 3.6982813, -4.3515210, 3.8294380

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0543227, upper bound: 1.0504137
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_A1_B1_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0528144, upper bound: 1.0504677
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_A1_A2

### Relational analysis result of IS_A1_B2_A1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0528144, upper bound: 1.0504677
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1730760, 0.6638508, -1.2651416, 2.5885944, -2.7616704, 1.9289924
1: -0.3801277, 0.8354926, -1.7519391, 2.8037324, -3.1838601, 2.5874317
2: -0.3072035, 0.9425610, -1.7190838, 3.1656418, -3.4728451, 2.6616449
3: -0.7048930, 0.9456613, -2.2556522, 3.6565268, -4.3614197, 3.2013135
4: -0.5783734, 1.1440246, -2.6154146, 3.6987345, -4.2771077, 3.7594392

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0485091, upper bound: 1.0374493
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0543227, upper bound: 1.0504076
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_A1_B1_A2_A1

### Relational analysis result of IS_A1_B2_A1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0528144, upper bound: 1.0504677
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_A2_A2

### Relational analysis result of IS_A1_B2_A1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0528144, upper bound: 1.0504677
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.2096419, 0.7101620, -1.2912724, 2.6481316, -2.8577733, 2.0014343
1: -0.4267260, 0.8983275, -1.7822843, 2.8719401, -3.2986660, 2.6806116
2: -0.3558276, 1.0129061, -1.7460456, 3.2356234, -3.5914507, 2.7589517
3: -0.7740620, 1.0201761, -2.2969935, 3.7360210, -4.5100827, 3.3171697
4: -0.6563050, 1.2186592, -2.6555572, 3.7709064, -4.4272113, 3.8742163

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0540514, upper bound: 1.0491135
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0436953, upper bound: 1.0420088
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1746585, 0.6661327, -1.2914166, 2.6484618, -2.8231201, 1.9575493
1: -0.3825408, 0.8379728, -1.7824845, 2.8723085, -3.2548492, 2.6204572
2: -0.3095305, 0.9459749, -1.7462265, 3.2360384, -3.5455689, 2.6922016
3: -0.7079530, 0.9492630, -2.2972631, 3.7364984, -4.4444509, 3.2465262
4: -0.5819045, 1.1485283, -2.6558306, 3.7713811, -4.3532858, 3.8043590

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0486733, upper bound: 1.0393975
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0544435, upper bound: 1.0491131
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0490575, upper bound: 1.0439313
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1806883, 0.6643587, -1.2609853, 2.5771461, -2.7578344, 1.9253440
1: -0.3907865, 0.8332261, -1.7460914, 2.7907536, -3.1815400, 2.5793176
2: -0.3165183, 0.9454506, -1.7142205, 3.1523950, -3.4689128, 2.6596711
3: -0.7177029, 0.9569116, -2.2473607, 3.6410317, -4.3587341, 3.2042723
4: -0.5945891, 1.1518157, -2.6084113, 3.6842589, -4.2788482, 3.7602270

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0519357, upper bound: 1.0503808
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0519357, upper bound: 1.0503808
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.1905538, 0.6793638, -1.2609853, 2.5771461, -2.7676997, 1.9403491
1: -0.4057388, 0.8529248, -1.7460914, 2.7907536, -3.1964922, 2.5990162
2: -0.3302413, 0.9696531, -1.7142205, 3.1523950, -3.4826362, 2.6838737
3: -0.7378664, 0.9809170, -2.2473607, 3.6410317, -4.3788977, 3.2282777
4: -0.6153858, 1.1839089, -2.6084113, 3.6842589, -4.2996445, 3.7923203

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0519357, upper bound: 1.0503808
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0519357, upper bound: 1.0503808
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.2122585, 0.7096910, -1.2661704, 2.5898392, -2.8020978, 1.9758613
1: -0.4381096, 0.8818882, -1.7478166, 2.8040323, -3.2421417, 2.6297047
2: -0.3631817, 1.0126708, -1.7160549, 3.1630225, -3.5262039, 2.7287257
3: -0.7779223, 1.0290604, -2.2481234, 3.6546252, -4.4325476, 3.2771838
4: -0.6656082, 1.2431200, -2.6115189, 3.6906371, -4.3562450, 3.8546388

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_A1

### Relational analysis result of IS_A1_B2_A1_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0457941, upper bound: 1.0439276
time: 0.47 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_B1

### Relational analysis result of IS_A1_B2_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0503540, upper bound: 1.0486297
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0503540, upper bound: 1.0486297
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.2122585, 0.7096910, -1.3370345, 2.6126482, -2.8249068, 2.0467255
1: -0.4381096, 0.8818882, -1.7949092, 2.8271534, -3.2652628, 2.6767974
2: -0.3631817, 1.0126708, -1.7629795, 3.2088265, -3.5720079, 2.7756503
3: -0.7779223, 1.0290604, -2.3119197, 3.7054596, -4.4833817, 3.3409801
4: -0.6656082, 1.2431200, -2.6737659, 3.7467427, -4.4123507, 3.9168859

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_A1

### Relational analysis result of IS_A1_B2_A1_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0457941, upper bound: 1.0439276
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0503540, upper bound: 1.0486761
time: 0.48 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0503540, upper bound: 1.0486761
time: 0.44 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.2683378, 0.8503048, -1.2491606, 2.5527794, -2.8211169, 2.0994654
1: -0.5085518, 1.0787544, -1.7293634, 2.7617764, -3.2703283, 2.8081174
2: -0.4326545, 1.2197943, -1.7005792, 3.1189947, -3.5516486, 2.9203732
3: -0.9030173, 1.2211714, -2.2224197, 3.6054430, -4.5084600, 3.4435911
4: -0.7851263, 1.4167428, -2.5882497, 3.6463084, -4.4314346, 4.0049925

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502910, upper bound: 1.0488995
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 31

Time for candidate selection: 5.57 seconds

### Candidate
type: A, layer: 3, pos: 31

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0464265, upper bound: 1.0463434
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504370, upper bound: 1.0492938
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0355247, upper bound: 1.0326395
time: 0.49 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0485168, upper bound: 1.0465429
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.2683378, 0.8503048, -1.2271219, 2.5771050, -2.8454428, 2.0774269
1: -0.5085518, 1.0787544, -1.6880672, 2.8331177, -3.3416693, 2.7668211
2: -0.4326545, 1.2197943, -1.6324024, 3.2002738, -3.6329279, 2.8521962
3: -0.9030173, 1.2211714, -2.2409055, 3.6114621, -4.5144796, 3.4620769
4: -0.7851263, 1.4167428, -2.5104861, 3.6776853, -4.4628115, 3.9272289

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502910, upper bound: 1.0488995
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 31

Time for candidate selection: 5.21 seconds

### Candidate
type: A, layer: 3, pos: 31

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0464265, upper bound: 1.0463434
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504370, upper bound: 1.0492938
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0355247, upper bound: 1.0326395
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_B2

### Relational analysis result of IS_A1_B2_A2_A1_B1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0485168, upper bound: 1.0465429
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.2644493, 0.8445802, -1.1650791, 2.3854909, -2.6499403, 2.0096593
1: -0.5032814, 1.0716369, -1.6274657, 2.6082609, -3.1115422, 2.6991024
2: -0.4271194, 1.2114094, -1.5950179, 2.9248562, -3.3519754, 2.8064268
3: -0.8964356, 1.2122335, -2.1051385, 3.3717260, -4.2681618, 3.3173714
4: -0.7770966, 1.4076097, -2.4340081, 3.4225998, -4.1996965, 3.8416173

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_A2_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0444293, upper bound: 1.0431251
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_A2_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0444293, upper bound: 1.0431251
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.2847042, 0.8710025, -1.2408764, 2.5350981, -2.8198023, 2.1118786
1: -0.5346295, 1.1044348, -1.7198091, 2.7445037, -3.2791331, 2.8242440
2: -0.4548847, 1.2482237, -1.6889806, 3.1038139, -3.5586987, 2.9372036
3: -0.9340510, 1.2600147, -2.2156267, 3.5820315, -4.5160823, 3.4756413
4: -0.8210288, 1.4488646, -2.5723977, 3.6292882, -4.4503169, 4.0212622

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505552, upper bound: 1.0499679
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505552, upper bound: 1.0499679
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.2971591, 0.8949012, -1.2408764, 2.5350981, -2.8322570, 2.1357772
1: -0.5497876, 1.1363558, -1.7198091, 2.7445037, -3.2942913, 2.8561649
2: -0.4700940, 1.2826568, -1.6889806, 3.1038139, -3.5739079, 2.9716368
3: -0.9585164, 1.2952633, -2.2156267, 3.5820315, -4.5405469, 3.5108900
4: -0.8436151, 1.4857011, -2.5723977, 3.6292882, -4.4729033, 4.0580988

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505552, upper bound: 1.0499679
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505552, upper bound: 1.0499679
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -1.2650044, 2.5882788, -0.2082714, 0.7079026, -1.9729069, 2.7965503
1: -1.7517481, 2.8033786, -0.4245980, 0.8958343, -2.6475825, 3.2279766
2: -1.7189114, 3.1652446, -0.3538412, 1.0096250, -2.7285364, 3.5190854
3: -2.2553968, 3.6560674, -0.7711517, 1.0166786, -3.2720754, 4.4272184
4: -2.6151581, 3.6982813, -0.6532400, 1.2142799, -3.8294380, 4.3515210

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504137, upper bound: 1.0543227
time: 0.38 seconds

## Relational analysis of IS_A2_B1_B1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B1_B1_A1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504677, upper bound: 1.0528144
time: 0.42 seconds

## Relational analysis of IS_A2_B1_B1_B1_A1_B1_B2

### Relational analysis result of IS_A2_B1_B1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504677, upper bound: 1.0543227
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -1.2651416, 2.5885944, -0.1730760, 0.6638508, -1.9289924, 2.7616704
1: -1.7519391, 2.8037324, -0.3801277, 0.8354926, -2.5874317, 3.1838601
2: -1.7190838, 3.1656418, -0.3072035, 0.9425610, -2.6616449, 3.4728448
3: -2.2556522, 3.6565268, -0.7048930, 0.9456613, -3.2013135, 4.3614197
4: -2.6154146, 3.6987345, -0.5783734, 1.1440246, -3.7594392, 4.2771077

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.1000000, mid=0.1000000, abs_max=1.1883488893508911
rel_dist={0: [-1.0558835892060525, 1.0558835892060525]}

## Binary search (step 1) starts
Candidate diff: 0.0500000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0540802, upper bound: 1.0510815
time: 0.39 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
time: 0.35 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.89 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.89
Output dim: 0, lower bound: -1.0540802, upper bound: 1.0510815
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.89
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.2352601, 0.7538407, -0.3035725, 0.8847764, -1.1200366, 1.0574131
1: -0.4706453, 0.9488738, -0.5660125, 1.0933844, -1.5640295, 1.5148864
2: -0.3915833, 1.0723588, -0.4826685, 1.2412479, -1.6328310, 1.5550274
3: -0.8320177, 1.0878556, -0.9617165, 1.2755736, -2.1075912, 2.0495720
4: -0.7029035, 1.3031529, -0.8354526, 1.4994075, -2.2023110, 2.1386056

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
time: 0.37 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
time: 0.36 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -1.3271873, 2.7002649, -0.3022430, 0.8821595, -2.2093468, 3.0025079
1: -1.8266034, 2.9257355, -0.5642669, 1.0904100, -2.9170134, 3.4900024
2: -1.7866864, 3.3100519, -0.4809870, 1.2378945, -3.0245810, 3.7910390
3: -2.3538351, 3.8103127, -0.9594972, 1.2719380, -3.6257727, 4.7698097
4: -2.7129741, 3.8588223, -0.8330401, 1.4958110, -4.2087851, 4.6918612

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
time: 0.36 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
time: 0.37 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.59 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.59
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.59
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.59
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.59
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.2352601, 0.7538407, -0.2352601, 0.7538407, -0.9891008, 0.9891006
1: -0.4706453, 0.9488738, -0.4706453, 0.9488738, -1.4195192, 1.4195192
2: -0.3915833, 1.0723588, -0.3915833, 1.0723588, -1.4639422, 1.4639422
3: -0.8320177, 1.0878556, -0.8320177, 1.0878556, -1.9198732, 1.9198732
4: -0.7029035, 1.3031529, -0.7029035, 1.3031529, -2.0060563, 2.0060563

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535217, upper bound: 1.0507277
time: 0.45 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0531956, upper bound: 1.0509582
time: 0.37 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.2352601, 0.7538407, -1.3271873, 2.7002649, -2.9355247, 2.0810280
1: -0.4706453, 0.9488738, -1.8266034, 2.9257355, -3.3963809, 2.7754772
2: -0.3915833, 1.0723588, -1.7866864, 3.3100519, -3.7016354, 2.8590453
3: -0.8320177, 1.0878556, -2.3538351, 3.8103127, -4.6423302, 3.4416907
4: -0.7029035, 1.3031529, -2.7129741, 3.8588223, -4.5617256, 4.0161266

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0536680, upper bound: 1.0509870
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0519334, upper bound: 1.0507646
time: 0.35 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -1.3271873, 2.7002649, -0.2352601, 0.7538407, -2.0810280, 2.9355249
1: -1.8266034, 2.9257355, -0.4706453, 0.9488738, -2.7754772, 3.3963809
2: -1.7866864, 3.3100519, -0.3915833, 1.0723588, -2.8590453, 3.7016351
3: -2.3538351, 3.8103127, -0.8320177, 1.0878556, -3.4416907, 4.6423302
4: -2.7129741, 3.8588223, -0.7029035, 1.3031529, -4.0161266, 4.5617256

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507166, upper bound: 1.0507511
time: 0.36 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507030, upper bound: 1.0507030
time: 0.37 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -1.3271873, 2.7002649, -1.3271873, 2.7002649, -4.0274525, 4.0274525
1: -1.8266034, 2.9257355, -1.8266034, 2.9257355, -4.7523389, 4.7523384
2: -1.7866864, 3.3100519, -1.7866864, 3.3100519, -5.0967383, 5.0967379
3: -2.3538351, 3.8103127, -2.3538351, 3.8103127, -6.1641479, 6.1641479
4: -2.7129741, 3.8588223, -2.7129741, 3.8588223, -6.5717964, 6.5717964

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0487934, upper bound: 1.0495110
time: 0.37 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506395, upper bound: 1.0506395
time: 0.43 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.41 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.41
Output dim: 0, lower bound: -1.0535217, upper bound: 1.0507277
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.41
Output dim: 0, lower bound: -1.0531956, upper bound: 1.0509582
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.41
Output dim: 0, lower bound: -1.0536680, upper bound: 1.0509870
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.41
Output dim: 0, lower bound: -1.0519334, upper bound: 1.0507646
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 2.41
Output dim: 0, lower bound: -1.0507166, upper bound: 1.0507511
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 2.41
Output dim: 0, lower bound: -1.0507030, upper bound: 1.0507030
IS_A2_B2_B1, status: Status.VERIFIED, split count: 3, time: 2.41
Output dim: 0, lower bound: -1.0487934, upper bound: 1.0495110
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 2.41
Output dim: 0, lower bound: -1.0506395, upper bound: 1.0506395

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.2013956, 0.7023641, -0.2304998, 0.7457045, -0.9471001, 0.9328639
1: -0.4206367, 0.8831170, -0.4635985, 0.9387211, -1.3593577, 1.3467155
2: -0.3466128, 1.0003821, -0.3852943, 1.0613912, -1.4080040, 1.3856764
3: -0.7602279, 1.0080743, -0.8218359, 1.0758421, -1.8360701, 1.8299102
4: -0.6370696, 1.2147777, -0.6937719, 1.2902606, -1.9273301, 1.9085495

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0534027, upper bound: 1.0514621
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A1_A2

### Relational analysis result of IS_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505944, upper bound: 1.0508893
time: 0.35 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.2363607, 0.7416377, -0.2258822, 0.7346829, -0.9710436, 0.9675199
1: -0.4722191, 0.9222932, -0.4567959, 0.9236316, -1.3958507, 1.3790891
2: -0.3964722, 1.0577438, -0.3797194, 1.0467503, -1.4432225, 1.4374632
3: -0.8244337, 1.0810699, -0.8123593, 1.0611860, -1.8856196, 1.8934293
4: -0.7141775, 1.3009543, -0.6861970, 1.2757928, -1.9899704, 1.9871514

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539857, upper bound: 1.0540170
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539857, upper bound: 1.0540170
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.2121267, 0.7201707, -1.3271873, 2.7002649, -2.9123917, 2.0473580
1: -0.4377390, 0.9047500, -1.8266034, 2.9257355, -3.3634744, 2.7313535
2: -0.3605663, 1.0264064, -1.7866864, 3.3100519, -3.6706183, 2.8130927
3: -0.7864621, 1.0343318, -2.3538351, 3.8103127, -4.5967746, 3.3881669
4: -0.6582627, 1.2474420, -2.7129741, 3.8588223, -4.5170841, 3.9604161

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0529761, upper bound: 1.0504533
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0529737, upper bound: 1.0508936
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.2998355, 0.9004637, -1.2732573, 2.5869870, -2.8868225, 2.1737208
1: -0.5571232, 1.1422836, -1.7557933, 2.8015742, -3.3586974, 2.8980770
2: -0.4746163, 1.2865998, -1.7188678, 3.1796603, -3.6542766, 3.0054674
3: -0.9696201, 1.2997241, -2.2681434, 3.6518369, -4.6214561, 3.5678666
4: -0.8469371, 1.4981039, -2.6162083, 3.7110577, -4.5579939, 4.1143122

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0503178, upper bound: 1.0491880
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0514709, upper bound: 1.0506707
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -1.3271873, 2.7002649, -0.2121267, 0.7201707, -2.0473580, 2.9123917
1: -1.8266034, 2.9257355, -0.4377390, 0.9047500, -2.7313535, 3.3634744
2: -1.7866864, 3.3100519, -0.3605663, 1.0264064, -2.8130927, 3.6706183
3: -2.3538351, 3.8103127, -0.7864621, 1.0343318, -3.3881669, 4.5967746
4: -2.7129741, 3.8588223, -0.6582627, 1.2474420, -3.9604161, 4.5170846

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504533, upper bound: 1.0529761
time: 0.33 seconds

## Relational analysis of IS_A2_B1_B1_B2

### Relational analysis result of IS_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508936, upper bound: 1.0530337
time: 0.44 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -1.2732573, 2.5869870, -0.2998355, 0.9004637, -2.1737208, 2.8868225
1: -1.7557933, 2.8015742, -0.5571232, 1.1422836, -2.8980770, 3.3586972
2: -1.7188678, 3.1796603, -0.4746163, 1.2865998, -3.0054674, 3.6542766
3: -2.2681434, 3.6518369, -0.9696201, 1.2997241, -3.5678668, 4.6214566
4: -2.6162083, 3.7110577, -0.8469371, 1.4981039, -4.1143122, 4.5579939

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0491880, upper bound: 1.0503178
time: 0.35 seconds

## Relational analysis of IS_A2_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506707, upper bound: 1.0514709
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -1.3112228, 2.6565771, -1.3654406, 2.6592798, -3.9705026, 4.0220175
1: -1.8038421, 2.8760159, -1.8352041, 2.8745258, -4.6783671, 4.7112198
2: -1.7678263, 3.2591543, -1.7965705, 3.2761538, -5.0439796, 5.0557246
3: -2.3217871, 3.7515779, -2.3616958, 3.7730355, -6.0948219, 6.1132736
4: -2.6856859, 3.8030729, -2.7261250, 3.8257360, -6.5114222, 6.5291967

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_B2_B1

### Relational analysis result of IS_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506227, upper bound: 1.0506259
time: 0.45 seconds

## Relational analysis of IS_A2_B2_B2_B2

### Relational analysis result of IS_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506091, upper bound: 1.0506091
time: 0.40 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.45 seconds
IS_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 2.45
Output dim: 0, lower bound: -1.0534027, upper bound: 1.0514621
IS_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 2.45
Output dim: 0, lower bound: -1.0505944, upper bound: 1.0508893
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.45
Output dim: 0, lower bound: -1.0539857, upper bound: 1.0540170
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.45
Output dim: 0, lower bound: -1.0539857, upper bound: 1.0540170
IS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 2.45
Output dim: 0, lower bound: -1.0529761, upper bound: 1.0504533
IS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 2.45
Output dim: 0, lower bound: -1.0529737, upper bound: 1.0508936
IS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 2.45
Output dim: 0, lower bound: -1.0503178, upper bound: 1.0491880
IS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 2.45
Output dim: 0, lower bound: -1.0514709, upper bound: 1.0506707
IS_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.45
Output dim: 0, lower bound: -1.0504533, upper bound: 1.0529761
IS_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.45
Output dim: 0, lower bound: -1.0508936, upper bound: 1.0530337
IS_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.45
Output dim: 0, lower bound: -1.0491880, upper bound: 1.0503178
IS_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.45
Output dim: 0, lower bound: -1.0506707, upper bound: 1.0514709
IS_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.45
Output dim: 0, lower bound: -1.0506227, upper bound: 1.0506259
IS_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.45
Output dim: 0, lower bound: -1.0506091, upper bound: 1.0506091

## BFS IS instance: IS_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.1851257, 0.6783313, -0.2304998, 0.7457045, -0.9308302, 0.9088311
1: -0.3969364, 0.8526834, -0.4635985, 0.9387211, -1.3356576, 1.3162818
2: -0.3242711, 0.9644121, -0.3852943, 1.0613912, -1.3856623, 1.3497064
3: -0.7251614, 0.9704387, -0.8218359, 1.0758421, -1.8010036, 1.7922746
4: -0.6038694, 1.1718525, -0.6937719, 1.2902606, -1.8941299, 1.8656244

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0526369, upper bound: 1.0509125
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0526369, upper bound: 1.0506646
time: 0.40 seconds

## BFS IS instance: IS_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.2674521, 0.8277271, -0.2095698, 0.7120004, -0.9794525, 1.0372969
1: -0.5075184, 1.0548140, -0.4301857, 0.8966833, -1.4042017, 1.4849997
2: -0.4327361, 1.2001319, -0.3549349, 1.0168408, -1.4495769, 1.5550668
3: -0.9017408, 1.1985904, -0.7846582, 1.0216281, -1.9233689, 1.9832486
4: -0.7864016, 1.4086894, -0.6497744, 1.2316995, -2.0181012, 2.0584638

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_A2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502823, upper bound: 1.0499613
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502524, upper bound: 1.0501089
time: 0.38 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.2363607, 0.7416377, -0.2013956, 0.7023641, -0.9387248, 0.9430333
1: -0.4722191, 0.9222932, -0.4206367, 0.8831170, -1.3553361, 1.3429298
2: -0.3964722, 1.0577438, -0.3466128, 1.0003821, -1.3968543, 1.4043566
3: -0.8244337, 1.0810699, -0.7602279, 1.0080743, -1.8325080, 1.8412979
4: -0.7141775, 1.3009543, -0.6370696, 1.2147777, -1.9289553, 1.9380239

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0522696, upper bound: 1.0508282
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508892, upper bound: 1.0505944
time: 0.38 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.2363607, 0.7416377, -0.2363607, 0.7416377, -0.9779984, 0.9779984
1: -0.4722191, 0.9222932, -0.4722191, 0.9222932, -1.3945123, 1.3945123
2: -0.3964722, 1.0577438, -0.3964722, 1.0577438, -1.4542160, 1.4542160
3: -0.8244337, 1.0810699, -0.8244337, 1.0810699, -1.9055036, 1.9055036
4: -0.7141775, 1.3009543, -0.7141775, 1.3009543, -2.0151320, 2.0151320

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0522696, upper bound: 1.0517318
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508893, upper bound: 1.0505944
time: 0.46 seconds

## BFS IS instance: IS_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.1851257, 0.6783313, -1.3238299, 2.6921263, -2.8772519, 2.0021613
1: -0.3969364, 0.8526834, -1.8218448, 2.9161830, -3.3131194, 2.6745281
2: -0.3242711, 0.9644121, -1.7826023, 3.2999420, -3.6242132, 2.7470145
3: -0.7251614, 0.9704387, -2.3470442, 3.7989707, -4.5241308, 3.3174829
4: -0.6038694, 1.1718525, -2.7069280, 3.8476067, -4.4514751, 3.8787804

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_A1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0520318, upper bound: 1.0500652
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0521033, upper bound: 1.0489605
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.2122585, 0.7096910, -1.3112228, 2.6565771, -2.8688357, 2.0209138
1: -0.4381096, 0.8818882, -1.8038421, 2.8760159, -3.3141253, 2.6857302
2: -0.3631817, 1.0126708, -1.7678263, 3.2591543, -3.6223359, 2.7804971
3: -0.7779223, 1.0290604, -2.3217871, 3.7515779, -4.5295000, 3.3508475
4: -0.6656082, 1.2431200, -2.6856859, 3.8030729, -4.4686813, 3.9288058

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_A2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513428, upper bound: 1.0487367
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513428, upper bound: 1.0508936
time: 0.47 seconds

## BFS IS instance: IS_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.2699254, 0.8529468, -1.2697539, 2.5781145, -2.8480399, 2.1227002
1: -0.5107998, 1.0817231, -1.7508032, 2.7910819, -3.3018813, 2.8325262
2: -0.4349182, 1.2233939, -1.7146187, 3.1685772, -3.6034954, 2.9380126
3: -0.9060124, 1.2251596, -2.2609606, 3.6395180, -4.5455303, 3.4861202
4: -0.7885253, 1.4212955, -2.6098983, 3.6988668, -4.4873919, 4.0311937

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_A1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0498831, upper bound: 1.0483126
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499064, upper bound: 1.0483442
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.3204554, 0.9300180, -1.2575243, 2.5452912, -2.8657465, 2.1875420
1: -0.5845242, 1.1713222, -1.7335773, 2.7542925, -3.3388164, 2.9048994
2: -0.5056426, 1.3326001, -1.7004819, 3.1311460, -3.6367886, 3.0330820
3: -1.0023541, 1.3492649, -2.2368600, 3.5958014, -4.5981555, 3.5861247
4: -0.8973903, 1.5542920, -2.5898132, 3.6576488, -4.5550389, 4.1441050

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499568, upper bound: 1.0485033
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2

### Relational analysis result of IS_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499568, upper bound: 1.0485033
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -1.3238299, 2.6921263, -0.1851257, 0.6783313, -2.0021613, 2.8772516
1: -1.8218448, 2.9161830, -0.3969364, 0.8526834, -2.6745281, 3.3131194
2: -1.7826023, 3.2999420, -0.3242711, 0.9644121, -2.7470145, 3.6242127
3: -2.3470442, 3.7989707, -0.7251614, 0.9704387, -3.3174829, 4.5241308
4: -2.7069280, 3.8476067, -0.6038694, 1.1718525, -3.8787804, 4.4514751

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500652, upper bound: 1.0520318
time: 0.38 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0489605, upper bound: 1.0521033
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -1.3112228, 2.6565771, -0.2122585, 0.7096910, -2.0209138, 2.8688357
1: -1.8038421, 2.8760159, -0.4381096, 0.8818882, -2.6857302, 3.3141253
2: -1.7678263, 3.2591543, -0.3631817, 1.0126708, -2.7804971, 3.6223359
3: -2.3217871, 3.7515779, -0.7779223, 1.0290604, -3.3508475, 4.5295000
4: -2.6856859, 3.8030729, -0.6656082, 1.2431200, -3.9288058, 4.4686809

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_B1_B2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0487367, upper bound: 1.0513428
time: 0.37 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0487367, upper bound: 1.0530337
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -1.2697539, 2.5781145, -0.2699254, 0.8529468, -2.1227005, 2.8480399
1: -1.7508032, 2.7910819, -0.5107998, 1.0817231, -2.8325262, 3.3018811
2: -1.7146187, 3.1685772, -0.4349182, 1.2233939, -2.9380126, 3.6034954
3: -2.2609606, 3.6395180, -0.9060124, 1.2251596, -3.4861202, 4.5455303
4: -2.6098983, 3.6988668, -0.7885253, 1.4212955, -4.0311937, 4.4873915

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B2_B1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0483126, upper bound: 1.0498831
time: 0.35 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0483442, upper bound: 1.0499064
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -1.2575243, 2.5452912, -0.3204554, 0.9300180, -2.1875422, 2.8657465
1: -1.7335773, 2.7542925, -0.5845242, 1.1713222, -2.9048991, 3.3388166
2: -1.7004819, 3.1311460, -0.5056426, 1.3326001, -3.0330820, 3.6367886
3: -2.2368600, 3.5958014, -1.0023541, 1.3492649, -3.5861247, 4.5981555
4: -2.5898132, 3.6576488, -0.8973903, 1.5542920, -4.1441050, 4.5550389

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_B2_B2_A1

### Relational analysis result of IS_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0485033, upper bound: 1.0499568
time: 0.39 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2

### Relational analysis result of IS_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0485033, upper bound: 1.0514709
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -1.3112228, 2.6565771, -1.3498325, 2.6225345, -3.9337573, 4.0064096
1: -1.8038421, 2.8760159, -1.8129959, 2.8317785, -4.6356201, 4.6890116
2: -1.7678263, 3.2591543, -1.7776842, 3.2286239, -4.9964504, 5.0368381
3: -2.3217871, 3.7515779, -2.3294129, 3.7206798, -6.0424666, 6.0809908
4: -2.6856859, 3.8030729, -2.6985631, 3.7726209, -6.4583068, 6.5016356

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474105, upper bound: 1.0482742
time: 0.40 seconds

## Relational analysis of IS_A2_B2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0474105, upper bound: 1.0506259
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -1.2575243, 2.5452912, -1.3900077, 2.7811232, -4.0386472, 3.9352989
1: -1.7335773, 2.7542925, -1.8534610, 3.0589395, -4.7925158, 4.6077528
2: -1.7004819, 3.1311460, -1.7888100, 3.4775665, -5.1780481, 4.9199553
3: -2.2368600, 3.5958014, -2.4571981, 3.9132597, -6.1501188, 6.0529995
4: -2.5898132, 3.6576488, -2.7346168, 3.9865932, -6.5764055, 6.3922653

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0472379, upper bound: 1.0482742
time: 0.39 seconds

## Relational analysis of IS_A2_B2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0472379, upper bound: 1.0506050
time: 0.38 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.53 seconds
IS_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -1.0526369, upper bound: 1.0509125
IS_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -1.0526369, upper bound: 1.0506646
IS_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -1.0502823, upper bound: 1.0499613
IS_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -1.0502524, upper bound: 1.0501089
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -1.0522696, upper bound: 1.0508282
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -1.0508892, upper bound: 1.0505944
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -1.0522696, upper bound: 1.0517318
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -1.0508893, upper bound: 1.0505944
IS_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -1.0520318, upper bound: 1.0500652
IS_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -1.0521033, upper bound: 1.0489605
IS_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -1.0513428, upper bound: 1.0487367
IS_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -1.0513428, upper bound: 1.0508936
IS_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -1.0498831, upper bound: 1.0483126
IS_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -1.0499064, upper bound: 1.0483442
IS_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -1.0499568, upper bound: 1.0485033
IS_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -1.0499568, upper bound: 1.0485033
IS_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -1.0500652, upper bound: 1.0520318
IS_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -1.0489605, upper bound: 1.0521033
IS_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -1.0487367, upper bound: 1.0513428
IS_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -1.0487367, upper bound: 1.0530337
IS_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -1.0483126, upper bound: 1.0498831
IS_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -1.0483442, upper bound: 1.0499064
IS_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -1.0485033, upper bound: 1.0499568
IS_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -1.0485033, upper bound: 1.0514709
IS_A2_B2_B2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.53
Output dim: 0, lower bound: -1.0474105, upper bound: 1.0482742
IS_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -1.0474105, upper bound: 1.0506259
IS_A2_B2_B2_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.53
Output dim: 0, lower bound: -1.0472379, upper bound: 1.0482742
IS_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -1.0472379, upper bound: 1.0506050

## BFS IS instance: IS_A1_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1774142, 0.6674564, -0.1957501, 0.6945465, -0.8719607, 0.8632065
1: -0.3853742, 0.8408388, -0.4116819, 0.8779179, -1.2632921, 1.2525207
2: -0.3130327, 0.9479907, -0.3352311, 0.9864889, -1.2995217, 1.2832217
3: -0.7106574, 0.9531208, -0.7560895, 0.9946122, -1.7052696, 1.7092104
4: -0.5868486, 1.1500280, -0.6179821, 1.1903985, -1.7772470, 1.7680101

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0511374, upper bound: 1.0504673
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0511374, upper bound: 1.0504832
time: 0.38 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1837167, 0.6764191, -0.2073033, 0.7129835, -0.8967003, 0.8837224
1: -0.3948803, 0.8508478, -0.4295223, 0.9057534, -1.3006337, 1.2803701
2: -0.3221599, 0.9616230, -0.3510356, 1.0140654, -1.3362253, 1.3126585
3: -0.7225855, 0.9674332, -0.7789927, 1.0250638, -1.7476492, 1.7464259
4: -0.6006362, 1.1680149, -0.6416922, 1.2259243, -1.8265605, 1.8097070

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0529365, upper bound: 1.0505067
time: 0.46 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0529365, upper bound: 1.0506646
time: 0.38 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.2593745, 0.8156036, -0.1769531, 0.6643244, -0.9236989, 0.9925568
1: -0.4961942, 1.0417651, -0.3817618, 0.8416852, -1.3378794, 1.4235269
2: -0.4210599, 1.1825309, -0.3056754, 0.9467642, -1.3678241, 1.4882064
3: -0.8866759, 1.1798939, -0.7219900, 0.9461035, -1.8327794, 1.9018838
4: -0.7686534, 1.3852768, -0.5748603, 1.1376491, -1.9063025, 1.9601371

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501900, upper bound: 1.0497300
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502823, upper bound: 1.0499488
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502823, upper bound: 1.0499488
time: 0.37 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.2659268, 0.8255982, -0.1860621, 0.6799164, -0.9458432, 1.0116602
1: -0.5052903, 1.0527209, -0.3954151, 0.8641713, -1.3694617, 1.4481360
2: -0.4304219, 1.1970170, -0.3199794, 0.9704835, -1.4009055, 1.5169964
3: -0.8989245, 1.1952195, -0.7411440, 0.9714044, -1.8703289, 1.9363635
4: -0.7828714, 1.4042351, -0.5964784, 1.1683257, -1.9511970, 2.0007136

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499663, upper bound: 1.0499663
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499663, upper bound: 1.0501089
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.2122585, 0.7096910, -0.2013956, 0.7023641, -0.9146227, 0.9110866
1: -0.4381096, 0.8818882, -0.4206367, 0.8831170, -1.3212266, 1.3025248
2: -0.3631817, 1.0126708, -0.3466128, 1.0003821, -1.3635638, 1.3592836
3: -0.7779223, 1.0290604, -0.7602279, 1.0080743, -1.7859967, 1.7892883
4: -0.6656082, 1.2431200, -0.6370696, 1.2147777, -1.8803859, 1.8801895

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0509164, upper bound: 1.0505128
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0511265, upper bound: 1.0505117
time: 0.38 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.3022564, 0.8574660, -0.1816620, 0.6706874, -0.9729438, 1.0391281
1: -0.5588838, 1.0796481, -0.3886601, 0.8448312, -1.4037150, 1.4683082
2: -0.4826456, 1.2362378, -0.3174605, 0.9567611, -1.4394066, 1.5536983
3: -0.9591278, 1.2640674, -0.7238576, 0.9572418, -1.9163697, 1.9879251
4: -0.8620855, 1.4785453, -0.5944541, 1.1581156, -2.0202012, 2.0729995

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499613, upper bound: 1.0502823
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501089, upper bound: 1.0502524
time: 0.46 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.2122585, 0.7096910, -0.2363607, 0.7416377, -0.9538963, 0.9460517
1: -0.4381096, 0.8818882, -0.4722191, 0.9222932, -1.3604028, 1.3541073
2: -0.3631817, 1.0126708, -0.3964722, 1.0577438, -1.4209255, 1.4091430
3: -0.7779223, 1.0290604, -0.8244337, 1.0810699, -1.8589923, 1.8534940
4: -0.6656082, 1.2431200, -0.7141775, 1.3009543, -1.9665625, 1.9572976

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504873, upper bound: 1.0502489
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0511265, upper bound: 1.0507046
time: 0.45 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.3022564, 0.8574660, -0.2196092, 0.7150588, -1.0173151, 1.0770752
1: -0.5588838, 1.0796481, -0.4448397, 0.8919341, -1.4508178, 1.5244879
2: -0.4826456, 1.2362378, -0.3713522, 1.0230851, -1.5057306, 1.6075900
3: -0.9591278, 1.2640674, -0.7955373, 1.0377979, -1.9969258, 2.0596046
4: -0.8620855, 1.4785453, -0.6779964, 1.2520282, -2.1141138, 2.1565418

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501958, upper bound: 1.0502418
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502167, upper bound: 1.0504029
time: 0.44 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1774142, 0.6674564, -1.2619333, 2.5806837, -2.7580976, 1.9293897
1: -0.3853742, 0.8408388, -1.7473245, 2.7944846, -3.1798587, 2.5881634
2: -0.3130327, 0.9479907, -1.7151394, 3.1558208, -3.4688535, 2.6631300
3: -0.7106574, 0.9531208, -2.2489846, 3.6454558, -4.3561134, 3.2021055
4: -0.5868486, 1.1500280, -2.6095755, 3.6878364, -4.2746849, 3.7596035

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507764, upper bound: 1.0489457
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507764, upper bound: 1.0489605
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1837167, 0.6764191, -1.2879562, 2.6403456, -2.8240623, 1.9643753
1: -0.3948803, 0.8508478, -1.7776947, 2.8628340, -3.2577140, 2.6285424
2: -0.3221599, 0.9616230, -1.7421074, 3.2259486, -3.5481083, 2.7037303
3: -0.7225855, 0.9674332, -2.2904172, 3.7252295, -4.4478149, 3.2578504
4: -0.6006362, 1.1680149, -2.6497891, 3.7601724, -4.3608079, 3.8178039

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0521033, upper bound: 1.0487686
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0472275, upper bound: 1.0435574
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.2122585, 0.7096910, -1.3023098, 2.6408219, -2.8530805, 2.0120008
1: -0.4381096, 0.8818882, -1.7915673, 2.8563752, -3.2944846, 2.6734555
2: -0.3631817, 1.0126708, -1.7563353, 3.2358289, -3.5990105, 2.7690060
3: -0.7779223, 1.0290604, -2.3039412, 3.7273993, -4.5053215, 3.3330016
4: -0.6656082, 1.2431200, -2.6680644, 3.7765968, -4.4422050, 3.9111843

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0463332, upper bound: 1.0438172
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499801, upper bound: 1.0484593
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501955, upper bound: 1.0484574
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.2122585, 0.7096910, -1.3654406, 2.6592798, -2.8715384, 2.0751317
1: -0.4381096, 0.8818882, -1.8352041, 2.8745258, -3.3126354, 2.7170923
2: -0.3631817, 1.0126708, -1.7965705, 3.2761538, -3.6393356, 2.8092413
3: -0.7779223, 1.0290604, -2.3616958, 3.7730355, -4.5509577, 3.3907561
4: -0.6656082, 1.2431200, -2.7261250, 3.8257360, -4.4913445, 3.9692450

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0463332, upper bound: 1.0438172
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499802, upper bound: 1.0484593
time: 0.45 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501955, upper bound: 1.0486730
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.2613858, 0.8388638, -1.2061714, 2.4643738, -2.7257595, 2.0450349
1: -0.4988431, 1.0658709, -1.6744742, 2.6667256, -3.1655688, 2.7403450
2: -0.4228014, 1.2040988, -1.6452420, 3.0215051, -3.4443061, 2.8493409
3: -0.8901179, 1.2038670, -2.1609187, 3.4823384, -4.3724566, 3.3647854
4: -0.7703297, 1.3968560, -2.5099316, 3.5356784, -4.3060079, 3.9067874

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0496968, upper bound: 1.0474992
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0498558, upper bound: 1.0483126
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0498558, upper bound: 1.0483126
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2683716, 0.8506285, -1.2340914, 2.5259178, -2.7942894, 2.0847199
1: -0.5085360, 1.0793722, -1.7064607, 2.7372730, -3.2458088, 2.7858329
2: -0.4325774, 1.2201154, -1.6737380, 3.0939741, -3.5265512, 2.8938534
3: -0.9031511, 1.2215762, -2.2043309, 3.5648692, -4.4680204, 3.4259071
4: -0.7849700, 1.4167454, -2.5520020, 3.6105902, -4.3955603, 3.9687474

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0496968, upper bound: 1.0480907
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0444293, upper bound: 1.0431251
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.3204554, 0.9300180, -1.2478310, 2.5246742, -2.8451293, 2.1778488
1: -0.5845242, 1.1713222, -1.7198234, 2.7284236, -3.3129478, 2.8911457
2: -0.5056426, 1.3326001, -1.6880045, 3.1017640, -3.6074066, 3.0206046
3: -1.0023541, 1.3492649, -2.2164874, 3.5648723, -4.5672264, 3.5657516
4: -0.8973903, 1.5542920, -2.5704732, 3.6249270, -4.5223174, 4.1247654

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0453070, upper bound: 1.0436244
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0490390, upper bound: 1.0482411
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0491765, upper bound: 1.0482103
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.3204554, 0.9300180, -1.3192331, 2.5616016, -2.8820570, 2.2492511
1: -0.5845242, 1.1713222, -1.7729726, 2.7681255, -3.3526497, 2.9442949
2: -0.5056426, 1.3326001, -1.7372446, 3.1637952, -3.6694379, 3.0698447
3: -1.0023541, 1.3492649, -2.2865701, 3.6362703, -4.6386242, 3.6358347
4: -0.8973903, 1.5542920, -2.6415663, 3.6967885, -4.5941782, 4.1958580

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0453071, upper bound: 1.0436244
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0490390, upper bound: 1.0489784
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0491765, upper bound: 1.0486524
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -1.2619333, 2.5806837, -0.1774142, 0.6674564, -1.9293897, 2.7580974
1: -1.7473245, 2.7944846, -0.3853742, 0.8408388, -2.5881634, 3.1798587
2: -1.7151394, 3.1558208, -0.3130327, 0.9479907, -2.6631300, 3.4688535
3: -2.2489846, 3.6454558, -0.7106574, 0.9531208, -3.2021055, 4.3561134
4: -2.6095755, 3.6878364, -0.5868486, 1.1500280, -3.7596035, 4.2746844

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B1_B1_A1_B1

### Relational analysis result of IS_A2_B1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0489457, upper bound: 1.0507764
time: 0.37 seconds

## Relational analysis of IS_A2_B1_B1_B1_A1_B2

### Relational analysis result of IS_A2_B1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0489457, upper bound: 1.0520318
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -1.2879562, 2.6403456, -0.1837167, 0.6764191, -1.9643753, 2.8240623
1: -1.7776947, 2.8628340, -0.3948803, 0.8508478, -2.6285424, 3.2577143
2: -1.7421074, 3.2259486, -0.3221599, 0.9616230, -2.7037303, 3.5481083
3: -2.2904172, 3.7252295, -0.7225855, 0.9674332, -3.2578504, 4.4478149
4: -2.6497891, 3.7601724, -0.6006362, 1.1680149, -3.8178039, 4.3608084

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0487686, upper bound: 1.0521033
time: 0.39 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A2

### Relational analysis result of IS_A2_B1_B1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0435574, upper bound: 1.0472275
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -1.3023098, 2.6408219, -0.2122585, 0.7096910, -2.0120008, 2.8530805
1: -1.7915673, 2.8563752, -0.4381096, 0.8818882, -2.6734555, 3.2944846
2: -1.7563353, 3.2358289, -0.3631817, 1.0126708, -2.7690060, 3.5990105
3: -2.3039412, 3.7273993, -0.7779223, 1.0290604, -3.3330016, 4.5053215
4: -2.6680644, 3.7765968, -0.6656082, 1.2431200, -3.9111843, 4.4422050

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0438172, upper bound: 1.0463332
time: 0.36 seconds

## Relational analysis of IS_A2_B1_B1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0484331, upper bound: 1.0499801
time: 0.40 seconds

## Relational analysis of IS_A2_B1_B1_B2_A1_B2

### Relational analysis result of IS_A2_B1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0484331, upper bound: 1.0501955
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -1.3654406, 2.6592798, -0.2122585, 0.7096910, -2.0751317, 2.8715384
1: -1.8352041, 2.8745258, -0.4381096, 0.8818882, -2.7170923, 3.3126349
2: -1.7965705, 3.2761538, -0.3631817, 1.0126708, -2.8092413, 3.6393352
3: -2.3616958, 3.7730355, -0.7779223, 1.0290604, -3.3907561, 4.5509577
4: -2.7261250, 3.8257360, -0.6656082, 1.2431200, -3.9692450, 4.4913440

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0438172, upper bound: 1.0463332
time: 0.36 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0484331, upper bound: 1.0513621
time: 0.39 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0484331, upper bound: 1.0510623
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -1.2061714, 2.4643738, -0.2613858, 0.8388638, -2.0450351, 2.7257595
1: -1.6744742, 2.6667256, -0.4988431, 1.0658709, -2.7403450, 3.1655688
2: -1.6452420, 3.0215051, -0.4228014, 1.2040988, -2.8493409, 3.4443064
3: -2.1609187, 3.4823384, -0.8901179, 1.2038670, -3.3647854, 4.3724566
4: -2.5099316, 3.5356784, -0.7703297, 1.3968560, -3.9067874, 4.3060079

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B2_B1_A1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0474992, upper bound: 1.0496968
time: 0.36 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_B2_B1_A1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0483126, upper bound: 1.0498558
time: 0.37 seconds

## Relational analysis of IS_A2_B1_B2_B1_A1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0483126, upper bound: 1.0498831
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -1.2340914, 2.5259178, -0.2683716, 0.8506285, -2.0847199, 2.7942894
1: -1.7064607, 2.7372730, -0.5085360, 1.0793722, -2.7858324, 3.2458091
2: -1.6737380, 3.0939741, -0.4325774, 1.2201154, -2.8938534, 3.5265510
3: -2.2043309, 3.5648692, -0.9031511, 1.2215762, -3.4259071, 4.4680204
4: -2.5520020, 3.6105902, -0.7849700, 1.4167454, -3.9687474, 4.3955603

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B2_B1_A2_A1

### Relational analysis result of IS_A2_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0480907, upper bound: 1.0497811
time: 0.39 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2_A2

### Relational analysis result of IS_A2_B1_B2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0431251, upper bound: 1.0444293
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -1.2478310, 2.5246742, -0.3204554, 0.9300180, -2.1778488, 2.8451293
1: -1.7198234, 2.7284236, -0.5845242, 1.1713222, -2.8911457, 3.3129478
2: -1.6880045, 3.1017640, -0.5056426, 1.3326001, -3.0206046, 3.6074066
3: -2.2164874, 3.5648723, -1.0023541, 1.3492649, -3.5657520, 4.5672264
4: -2.5704732, 3.6249270, -0.8973903, 1.5542920, -4.1247654, 4.5223174

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B2_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0434039, upper bound: 1.0453070
time: 0.36 seconds

## Relational analysis of IS_A2_B1_B2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_B2_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0481636, upper bound: 1.0499567
time: 0.40 seconds

## Relational analysis of IS_A2_B1_B2_B2_A1_A2

### Relational analysis result of IS_A2_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0481636, upper bound: 1.0499567
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -1.3192331, 2.5616016, -0.3204554, 0.9300180, -2.2492511, 2.8820570
1: -1.7729726, 2.7681255, -0.5845242, 1.1713222, -2.9442949, 3.3526497
2: -1.7372446, 3.1637952, -0.5056426, 1.3326001, -3.0698447, 3.6694379
3: -2.2865701, 3.6362703, -1.0023541, 1.3492649, -3.6358347, 4.6386242
4: -2.6415663, 3.6967885, -0.8973903, 1.5542920, -4.1958580, 4.5941787

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B2_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0434039, upper bound: 1.0447400
time: 0.43 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_B2_B2_A2_A1

### Relational analysis result of IS_A2_B1_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0481636, upper bound: 1.0493443
time: 0.39 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2_A2

### Relational analysis result of IS_A2_B1_B2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0481636, upper bound: 1.0493443
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -1.3654406, 2.6592798, -1.3498325, 2.6225345, -3.9879746, 4.0091124
1: -1.8352041, 2.8745258, -1.8129959, 2.8317785, -4.6669827, 4.6875219
2: -1.7965705, 3.2761538, -1.7776842, 3.2286239, -5.0251942, 5.0538373
3: -2.3616958, 3.7730355, -2.3294129, 3.7206798, -6.0823750, 6.1024485
4: -2.7261250, 3.8257360, -2.6985631, 3.7726209, -6.4987450, 6.5242987

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_B2_B1_A2_B1

### Relational analysis result of IS_A2_B2_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0426407, upper bound: 1.0431026
time: 0.38 seconds

## Relational analysis of IS_A2_B2_B2_B1_A2_B2

### Relational analysis result of IS_A2_B2_B2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0257779, upper bound: 1.0273470
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -1.3192331, 2.5616016, -1.3900077, 2.7811232, -4.1003561, 3.9516094
1: -1.7729726, 2.7681255, -1.8534610, 3.0589395, -4.8319116, 4.6215863
2: -1.7372446, 3.1637952, -1.7888100, 3.4775665, -5.2148113, 4.9526043
3: -2.2865701, 3.6362703, -2.4571981, 3.9132597, -6.1998301, 6.0934682
4: -2.6415663, 3.6967885, -2.7346168, 3.9865932, -6.6281595, 6.4314051

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_B2_B2_A2_B1

### Relational analysis result of IS_A2_B2_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0424720, upper bound: 1.0426097
time: 0.39 seconds

## Relational analysis of IS_A2_B2_B2_B2_A2_B2

### Relational analysis result of IS_A2_B2_B2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0379121, upper bound: 1.0379121
time: 0.40 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.81 seconds
IS_A1_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0511374, upper bound: 1.0504673
IS_A1_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0511374, upper bound: 1.0504832
IS_A1_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0529365, upper bound: 1.0505067
IS_A1_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0529365, upper bound: 1.0506646
IS_A1_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0502823, upper bound: 1.0499488
IS_A1_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0502823, upper bound: 1.0499488
IS_A1_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0499663, upper bound: 1.0499663
IS_A1_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0499663, upper bound: 1.0501089
IS_A1_B1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0509164, upper bound: 1.0505128
IS_A1_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0511265, upper bound: 1.0505117
IS_A1_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0499613, upper bound: 1.0502823
IS_A1_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0501089, upper bound: 1.0502524
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0504873, upper bound: 1.0502489
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0511265, upper bound: 1.0507046
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0501958, upper bound: 1.0502418
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0502167, upper bound: 1.0504029
IS_A1_B2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0507764, upper bound: 1.0489457
IS_A1_B2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0507764, upper bound: 1.0489605
IS_A1_B2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0521033, upper bound: 1.0487686
IS_A1_B2_A1_A1_B2_B2, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0472275, upper bound: 1.0435574
IS_A1_B2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0499801, upper bound: 1.0484593
IS_A1_B2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0501955, upper bound: 1.0484574
IS_A1_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0499802, upper bound: 1.0484593
IS_A1_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0501955, upper bound: 1.0486730
IS_A1_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0498558, upper bound: 1.0483126
IS_A1_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0498558, upper bound: 1.0483126
IS_A1_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0496968, upper bound: 1.0480907
IS_A1_B2_A2_A1_B2_B2, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0444293, upper bound: 1.0431251
IS_A1_B2_A2_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0490390, upper bound: 1.0482411
IS_A1_B2_A2_A2_B1_A2, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0491765, upper bound: 1.0482103
IS_A1_B2_A2_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0490390, upper bound: 1.0489784
IS_A1_B2_A2_A2_B2_A2, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0491765, upper bound: 1.0486524
IS_A2_B1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0489457, upper bound: 1.0507764
IS_A2_B1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0489457, upper bound: 1.0520318
IS_A2_B1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0487686, upper bound: 1.0521033
IS_A2_B1_B1_B1_A2_A2, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0435574, upper bound: 1.0472275
IS_A2_B1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0484331, upper bound: 1.0499801
IS_A2_B1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0484331, upper bound: 1.0501955
IS_A2_B1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0484331, upper bound: 1.0513621
IS_A2_B1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0484331, upper bound: 1.0510623
IS_A2_B1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0483126, upper bound: 1.0498558
IS_A2_B1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0483126, upper bound: 1.0498831
IS_A2_B1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0480907, upper bound: 1.0497811
IS_A2_B1_B2_B1_A2_A2, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0431251, upper bound: 1.0444293
IS_A2_B1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0481636, upper bound: 1.0499567
IS_A2_B1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0481636, upper bound: 1.0499567
IS_A2_B1_B2_B2_A2_A1, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0481636, upper bound: 1.0493443
IS_A2_B1_B2_B2_A2_A2, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0481636, upper bound: 1.0493443
IS_A2_B2_B2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0426407, upper bound: 1.0431026
IS_A2_B2_B2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0257779, upper bound: 1.0273470
IS_A2_B2_B2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0424720, upper bound: 1.0426097
IS_A2_B2_B2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.81
Output dim: 0, lower bound: -1.0379121, upper bound: 1.0379121

## BFS IS instance: IS_A1_B1_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1546751, 0.6344486, -0.1957501, 0.6945465, -0.8492216, 0.8301988
1: -0.3508692, 0.8027984, -0.4116819, 0.8779179, -1.2287872, 1.2144803
2: -0.2792361, 0.8971295, -0.3352311, 0.9864889, -1.2657250, 1.2323606
3: -0.6675518, 0.9013090, -0.7560895, 0.9946122, -1.6621640, 1.6573985
4: -0.5352525, 1.0835073, -0.6179821, 1.1903985, -1.7256509, 1.7014894

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0517257, upper bound: 1.0509037
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0517257, upper bound: 1.0509037
time: 0.44 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1633435, 0.6488099, -0.1957501, 0.6945465, -0.8578900, 0.8445600
1: -0.3648036, 0.8238323, -0.4116819, 0.8779179, -1.2427216, 1.2355142
2: -0.2912067, 0.9218535, -0.3352311, 0.9864889, -1.2776957, 1.2570846
3: -0.6862502, 0.9238169, -0.7560895, 0.9946122, -1.6808624, 1.6799064
4: -0.5532730, 1.1132669, -0.6179821, 1.1903985, -1.7436714, 1.7312491

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0517257, upper bound: 1.0509125
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0517257, upper bound: 1.0509125
time: 0.35 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.1837167, 0.6764191, -0.1796174, 0.6729426, -0.8566594, 0.8560364
1: -0.3948803, 0.8508478, -0.3886608, 0.8546216, -1.2495019, 1.2395086
2: -0.3221599, 0.9616230, -0.3137678, 0.9577398, -1.2798996, 1.2753909
3: -0.7225855, 0.9674332, -0.7210382, 0.9618257, -1.6844113, 1.6884713
4: -0.6006362, 1.1680149, -0.5868647, 1.1561475, -1.7567837, 1.7548796

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_A1_B2_B1_B1

### Relational analysis result of IS_A1_B1_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0529365, upper bound: 1.0505067
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_B1_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0529365, upper bound: 1.0505067
time: 0.49 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.1837167, 0.6764191, -0.2148956, 0.7115479, -0.8952646, 0.8913147
1: -0.3948803, 0.8508478, -0.4401824, 0.8935100, -1.2883904, 1.2910303
2: -0.3221599, 0.9616230, -0.3639436, 1.0145798, -1.3367397, 1.3255665
3: -0.7225855, 0.9674332, -0.7847203, 1.0327761, -1.7553616, 1.7521534
4: -0.6006362, 1.1680149, -0.6645471, 1.2404692, -1.8411055, 1.8325620

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_A1_B2_B2_B1

### Relational analysis result of IS_A1_B1_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0529365, upper bound: 1.0505082
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_A1_B2_B2_B1

### Relational analysis result of IS_A1_B1_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0529365, upper bound: 1.0506646
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_B2_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0529365, upper bound: 1.0506646
time: 0.42 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.2593745, 0.8156036, -0.1761124, 0.6674893, -0.9268638, 0.9917160
1: -0.4961942, 1.0417651, -0.3832366, 0.8439143, -1.3401084, 1.4250017
2: -0.4210599, 1.1825309, -0.3080643, 0.9467945, -1.3678544, 1.4905952
3: -0.8866759, 1.1798939, -0.7159681, 0.9515759, -1.8382518, 1.8958620
4: -0.7686534, 1.3852768, -0.5781275, 1.1423141, -1.9109675, 1.9634043

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501900, upper bound: 1.0496214
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4

Time for candidate selection: 5.09 seconds

### Candidate
type: A, layer: 3, pos: 31

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0464128, upper bound: 1.0469908
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B1_B2

### Relational analysis result of IS_A1_B1_A1_A2_B1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0463702, upper bound: 1.0461835
time: 0.37 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.2593745, 0.8156036, -0.2453578, 0.7792019, -1.0385764, 1.0609614
1: -0.4961942, 1.0417651, -0.4794726, 0.9961535, -1.4923477, 1.5212376
2: -0.4210599, 1.1825309, -0.3997424, 1.1196429, -1.5407028, 1.5822732
3: -0.8866759, 1.1798939, -0.8602813, 1.1326797, -2.0193555, 2.0401752
4: -0.7686534, 1.3852768, -0.7337275, 1.3247181, -2.0933714, 2.1190042

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501900, upper bound: 1.0496214
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 11

Time for candidate selection: 4.95 seconds

### Candidate
type: A, layer: 3, pos: 31

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0464128, upper bound: 1.0466616
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A1_A2_B1_B2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0463466, upper bound: 1.0465223
time: 0.40 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.2659268, 0.8255982, -0.1594403, 0.6406519, -0.9065788, 0.9850385
1: -0.5052903, 1.0527209, -0.3558388, 0.8156011, -1.3208914, 1.4085597
2: -0.4304219, 1.1970170, -0.2836999, 0.9132303, -1.3436522, 1.4807168
3: -0.8989245, 1.1952195, -0.6835538, 0.9105985, -1.8095231, 1.8787732
4: -0.7828714, 1.4042351, -0.5426459, 1.0981200, -1.8809913, 1.9468811

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_B1

### Relational analysis result of IS_A1_B1_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499216, upper bound: 1.0498500
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_B1

### Relational analysis result of IS_A1_B1_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499663, upper bound: 1.0499663
time: 0.45 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_B1_B2

### Relational analysis result of IS_A1_B1_A1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499663, upper bound: 1.0499663
time: 0.44 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.2659268, 0.8255982, -0.1979399, 0.6845874, -0.9505142, 1.0235381
1: -0.5052903, 1.0527209, -0.4124926, 0.8628027, -1.3680930, 1.4652135
2: -0.4304219, 1.1970170, -0.3384907, 0.9797636, -1.4101856, 1.5355077
3: -0.8989245, 1.1952195, -0.7554644, 0.9895242, -1.8884487, 1.9506840
4: -0.7828714, 1.4042351, -0.6275674, 1.1925520, -1.9754233, 2.0318027

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499216, upper bound: 1.0500052
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499663, upper bound: 1.0501089
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2_B2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499663, upper bound: 1.0501089
time: 0.47 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.1806883, 0.6643587, -0.1938861, 0.6916971, -0.8723854, 0.8582449
1: -0.3907865, 0.8332261, -0.4094110, 0.8715558, -1.2623423, 1.2426372
2: -0.3165183, 0.9454506, -0.3356451, 0.9841505, -1.3006688, 1.2810957
3: -0.7177029, 0.9569116, -0.7461731, 0.9910460, -1.7087488, 1.7030847
4: -0.5945891, 1.1518157, -0.6204394, 1.1932946, -1.7878838, 1.7722551

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A1_A1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0496573, upper bound: 1.0503994
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0509164, upper bound: 1.0505128
time: 0.45 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0509164, upper bound: 1.0505128
time: 0.41 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.1905538, 0.6793638, -0.1999941, 0.7004576, -0.8910114, 0.8793579
1: -0.4057388, 0.8529248, -0.4185898, 0.8813001, -1.2870389, 1.2715147
2: -0.3302413, 0.9696531, -0.3445140, 0.9975934, -1.3278346, 1.3141670
3: -0.7378664, 0.9809170, -0.7576658, 1.0050879, -1.7429543, 1.7385828
4: -0.6153858, 1.1839089, -0.6338574, 1.2109582, -1.8263440, 1.8177664

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A1_A2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0496573, upper bound: 1.0504078
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0511265, upper bound: 1.0505117
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0511265, upper bound: 1.0505117
time: 0.40 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.2689438, 0.8082717, -0.1743314, 0.6604250, -0.9293687, 0.9826031
1: -0.5119697, 1.0272677, -0.3777814, 0.8338645, -1.3458343, 1.4050491
2: -0.4340174, 1.1612520, -0.3066375, 0.9414527, -1.3754702, 1.4678895
3: -0.8960808, 1.1860839, -0.7104456, 0.9413403, -1.8374211, 1.8965294
4: -0.7881429, 1.3786812, -0.5781132, 1.1373857, -1.9255285, 1.9567944

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A2_A1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0497300, upper bound: 1.0501900
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499613, upper bound: 1.0502823
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499613, upper bound: 1.0502823
time: 0.43 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.2798769, 0.8257407, -0.1802223, 0.6687382, -0.9486152, 1.0059630
1: -0.5248469, 1.0494590, -0.3865340, 0.8429638, -1.3678107, 1.4359930
2: -0.4479707, 1.1899524, -0.3153016, 0.9539230, -1.4018937, 1.5052540
3: -0.9173980, 1.2125832, -0.7211983, 0.9542155, -1.8716135, 1.9337814
4: -0.8094229, 1.4129685, -0.5911303, 1.1542197, -1.9636426, 2.0040989

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A2_A2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500052, upper bound: 1.0501764
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501089, upper bound: 1.0502524
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501089, upper bound: 1.0502524
time: 0.43 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.2041048, 0.6977630, -0.2045429, 0.6960640, -0.9001688, 0.9023059
1: -0.4260215, 0.8692849, -0.4246404, 0.8731498, -1.2991712, 1.2939253
2: -0.3512143, 0.9951268, -0.3495150, 0.9898366, -1.3410510, 1.3446417
3: -0.7626776, 1.0101199, -0.7641593, 1.0075936, -1.7702712, 1.7742791
4: -0.6474289, 1.2196317, -0.6426289, 1.2076913, -1.8551202, 1.8622606

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504873, upper bound: 1.0502489
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504873, upper bound: 1.0502489
time: 0.45 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2108487, 0.7076982, -0.2148956, 0.7115479, -0.9223966, 0.9225938
1: -0.4360059, 0.8799497, -0.4401824, 0.8935100, -1.3295159, 1.3201323
2: -0.3610466, 1.0098436, -0.3639436, 1.0145798, -1.3756263, 1.3737872
3: -0.7752968, 1.0258479, -0.7847203, 1.0327761, -1.8080729, 1.8105682
4: -0.6623644, 1.2392564, -0.6645471, 1.2404692, -1.9028336, 1.9038036

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0511128, upper bound: 1.0507046
time: 0.49 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0509164, upper bound: 1.0507046
time: 0.45 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.2941689, 0.8452820, -0.1890319, 0.6695676, -0.9637365, 1.0343139
1: -0.5476846, 1.0665569, -0.3991820, 0.8432879, -1.3909724, 1.4657389
2: -0.4709241, 1.2176194, -0.3245404, 0.9557487, -1.4266729, 1.5421598
3: -0.9439932, 1.2445780, -0.7355406, 0.9653443, -1.9093375, 1.9801186
4: -0.8442773, 1.4537967, -0.6068336, 1.1604218, -2.0046992, 2.0606303

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501823, upper bound: 1.0501999
time: 0.46 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501823, upper bound: 1.0501999
time: 0.44 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.3007836, 0.8553913, -0.1979399, 0.6845874, -0.9853710, 1.0533313
1: -0.5566587, 1.0776916, -0.4124926, 0.8628027, -1.4194614, 1.4901842
2: -0.4803994, 1.2331573, -0.3384907, 0.9797636, -1.4601630, 1.5716480
3: -0.9563999, 1.2606578, -0.7554644, 0.9895242, -1.9459240, 2.0161223
4: -0.8586895, 1.4743117, -0.6275674, 1.1925520, -2.0512414, 2.1018791

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501824, upper bound: 1.0504029
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501824, upper bound: 1.0504029
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1546751, 0.6344486, -1.2619333, 2.5806837, -2.7353587, 1.8963820
1: -0.3508692, 0.8027984, -1.7473245, 2.7944846, -3.1453536, 2.5501227
2: -0.2792361, 0.8971295, -1.7151394, 3.1558208, -3.4350569, 2.6122689
3: -0.6675518, 0.9013090, -2.2489846, 3.6454558, -4.3130074, 3.1502936
4: -0.5352525, 1.0835073, -2.6095755, 3.6878364, -4.2230887, 3.6930828

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0516579, upper bound: 1.0500071
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0516579, upper bound: 1.0500071
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1633435, 0.6488099, -1.2619333, 2.5806837, -2.7440271, 1.9107432
1: -0.3648036, 0.8238323, -1.7473245, 2.7944846, -3.1592882, 2.5711567
2: -0.2912067, 0.9218535, -1.7151394, 3.1558208, -3.4470277, 2.6369929
3: -0.6862502, 0.9238169, -2.2489846, 3.6454558, -4.3317060, 3.1728015
4: -0.5532730, 1.1132669, -2.6095755, 3.6878364, -4.2411094, 3.7228425

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0516579, upper bound: 1.0500652
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0516579, upper bound: 1.0500652
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.1662051, 0.6552466, -1.1826556, 2.4220457, -2.5882504, 1.8379022
1: -0.3697689, 0.8258541, -1.6509128, 2.6472325, -3.0170014, 2.4767668
2: -0.2970747, 0.9300112, -1.6173604, 2.9671931, -3.2642674, 2.5473716
3: -0.6928098, 0.9316384, -2.1338735, 3.4227364, -4.1155457, 3.0655117
4: -0.5633935, 1.1270266, -2.4657319, 3.4712639, -4.0346575, 3.5927584

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_A1_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0458322, upper bound: 1.0435574
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_A1_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0458322, upper bound: 1.0435574
time: 0.49 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1806883, 0.6643587, -1.2865010, 2.6105459, -2.7912340, 1.9508598
1: -0.3907865, 0.8332261, -1.7714832, 2.8234062, -3.2141926, 2.6047094
2: -0.3165183, 0.9454506, -1.7385111, 3.1969161, -3.5134342, 2.6839616
3: -0.7177029, 0.9569116, -2.2768168, 3.6857088, -4.4034109, 3.2337284
4: -0.5945891, 1.1518157, -2.6420763, 3.7331512, -4.3277402, 3.7938919

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A2_B1_A1_A1

### Relational analysis result of IS_A1_B2_A1_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0488306, upper bound: 1.0482558
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499801, upper bound: 1.0484593
time: 0.44 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499801, upper bound: 1.0484593
time: 0.45 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.1905538, 0.6793638, -1.2995594, 2.6373627, -2.8279166, 1.9789232
1: -0.4057388, 0.8529248, -1.7885118, 2.8528142, -3.2585526, 2.6414366
2: -0.3302413, 0.9696531, -1.7535458, 3.2309113, -3.5611525, 2.7231989
3: -0.7378664, 0.9809170, -2.3000088, 3.7223716, -4.4602375, 3.2809258
4: -0.6153858, 1.1839089, -2.6641552, 3.7707973, -4.3861828, 3.8480642

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A2_B1_A2_A1

### Relational analysis result of IS_A1_B2_A1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499380, upper bound: 1.0482623
time: 0.47 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501955, upper bound: 1.0484574
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501955, upper bound: 1.0484574
time: 0.55 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1806883, 0.6643587, -1.3428411, 2.6253977, -2.8060856, 2.0071998
1: -0.3907865, 0.8332261, -1.8112683, 2.8375440, -3.2283304, 2.6444945
2: -0.3165183, 0.9454506, -1.7757163, 3.2324891, -3.5490069, 2.7211668
3: -0.7177029, 0.9569116, -2.3288047, 3.7250857, -4.4427881, 3.2857163
4: -0.5945891, 1.1518157, -2.6959231, 3.7771926, -4.3717818, 3.8477387

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A2_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0436815, upper bound: 1.0435448
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0497719, upper bound: 1.0486099
time: 0.48 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0497720, upper bound: 1.0486730
time: 0.45 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.1905538, 0.6793638, -1.3636296, 2.6562603, -2.8468142, 2.0429935
1: -0.4057388, 0.8529248, -1.8326011, 2.8714428, -3.2771814, 2.6855259
2: -0.3302413, 0.9696531, -1.7941780, 3.2718086, -3.6020498, 2.7638311
3: -0.7378664, 0.9809170, -2.3584447, 3.7686601, -4.5065260, 3.3393617
4: -0.6153858, 1.1839089, -2.7227578, 3.8206248, -4.4360104, 3.9066668

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0497720, upper bound: 1.0486099
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0497720, upper bound: 1.0486730
time: 0.47 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.2613858, 0.8388638, -1.2458898, 2.5447118, -2.8060973, 2.0847530
1: -0.4988431, 1.0658709, -1.7246733, 2.7523551, -3.2511978, 2.7905440
2: -0.4228014, 1.2040988, -1.6965399, 3.1089749, -3.5317764, 2.9006386
3: -0.8901179, 1.2038670, -2.2156568, 3.5941498, -4.4842672, 3.4195237
4: -0.7703297, 1.3968560, -2.5822523, 3.6352324, -4.4055614, 3.9791083

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0496940, upper bound: 1.0470154
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 31

Time for candidate selection: 5.11 seconds

### Candidate
type: A, layer: 3, pos: 31

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0460101, upper bound: 1.0453630
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0498558, upper bound: 1.0483126
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: A, layer: 5, pos: 40
type: B, layer: 5, pos: 40
type: B, layer: 5, pos: 22
type: A, layer: 5, pos: 22
type: A, layer: 5, pos: 33
type: A, layer: 5, pos: 35
type: B, layer: 5, pos: 18
type: A, layer: 5, pos: 34
type: A, layer: 5, pos: 41
type: B, layer: 5, pos: 35
type: A, layer: 5, pos: 20
type: B, layer: 5, pos: 20
type: B, layer: 5, pos: 0
type: B, layer: 5, pos: 16
type: A, layer: 5, pos: 18
type: A, layer: 5, pos: 21
type: A, layer: 5, pos: 13
type: B, layer: 5, pos: 9
type: B, layer: 5, pos: 33
type: B, layer: 5, pos: 42
type: A, layer: 5, pos: 27

Time for candidate selection: 10.76 seconds

### Candidate
type: A, layer: 5, pos: 40

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 40

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0373547, upper bound: 1.0380098
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 22

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 22

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_A1

### Relational analysis result of IS_A1_B2_A2_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0450647, upper bound: 1.0424943
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_B1_A2

### Relational analysis result of IS_A1_B2_A2_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0430154, upper bound: 1.0422080
time: 0.47 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.2613858, 0.8388638, -1.2228512, 2.5676465, -2.8290319, 2.0617146
1: -0.4988431, 1.0658709, -1.6822026, 2.8219018, -3.3207450, 2.7480736
2: -0.4228014, 1.2040988, -1.6273031, 3.1887884, -3.6115899, 2.8314018
3: -0.8901179, 1.2038670, -2.2329960, 3.5981042, -4.4882221, 3.4368629
4: -0.7703297, 1.3968560, -2.5031996, 3.6648788, -4.4352083, 3.9000554

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0496940, upper bound: 1.0470154
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 31

Time for candidate selection: 5.15 seconds

### Candidate
type: A, layer: 3, pos: 31

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0460101, upper bound: 1.0453630
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0498558, upper bound: 1.0483126
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: B, layer: 5, pos: 40
type: B, layer: 5, pos: 22
type: A, layer: 5, pos: 40
type: A, layer: 5, pos: 22
type: A, layer: 5, pos: 34
type: B, layer: 5, pos: 34
type: A, layer: 5, pos: 33
type: A, layer: 5, pos: 35
type: B, layer: 5, pos: 21
type: A, layer: 5, pos: 41
type: A, layer: 5, pos: 21
type: B, layer: 5, pos: 20
type: A, layer: 5, pos: 18
type: B, layer: 5, pos: 18
type: A, layer: 5, pos: 20
type: B, layer: 5, pos: 35
type: B, layer: 5, pos: 0
type: B, layer: 5, pos: 41
type: B, layer: 5, pos: 16
type: B, layer: 5, pos: 33
type: A, layer: 5, pos: 13
type: B, layer: 5, pos: 9
type: B, layer: 5, pos: 42
type: A, layer: 5, pos: 27

Time for candidate selection: 11.61 seconds

### Candidate
type: B, layer: 5, pos: 40

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0373547, upper bound: 1.0380098
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 22

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 40

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 22

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_A1

### Relational analysis result of IS_A1_B2_A2_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0450647, upper bound: 1.0424943
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_B2_A2

### Relational analysis result of IS_A1_B2_A2_A1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0430154, upper bound: 1.0422080
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.2416445, 0.8100984, -1.1263976, 2.3026717, -2.5443163, 1.9364957
1: -0.4727030, 1.0302428, -1.5756736, 2.5174496, -2.9901526, 2.6059160
2: -0.3940786, 1.1623082, -1.5464408, 2.8280895, -3.2221680, 2.7087483
3: -0.8569319, 1.1583879, -2.0403800, 3.2564921, -4.1134238, 3.1987677
4: -0.7284358, 1.3506840, -2.3646054, 3.3128293, -4.0412650, 3.7152896

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_A2_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0444293, upper bound: 1.0431251
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_A2_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0444293, upper bound: 1.0431251
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -1.2619333, 2.5806837, -0.1546751, 0.6344486, -1.8963820, 2.7353587
1: -1.7473245, 2.7944846, -0.3508692, 0.8027984, -2.5501227, 3.1453538
2: -1.7151394, 3.1558208, -0.2792361, 0.8971295, -2.6122689, 3.4350569
3: -2.2489846, 3.6454558, -0.6675518, 0.9013090, -3.1502936, 4.3130069
4: -2.6095755, 3.6878364, -0.5352525, 1.0835073, -3.6930828, 4.2230887

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_B1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500071, upper bound: 1.0516579
time: 0.42 seconds

## Relational analysis of IS_A2_B1_B1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500071, upper bound: 1.0516579
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -1.2619333, 2.5806837, -0.1633435, 0.6488099, -1.9107432, 2.7440269
1: -1.7473245, 2.7944846, -0.3648036, 0.8238323, -2.5711567, 3.1592882
2: -1.7151394, 3.1558208, -0.2912067, 0.9218535, -2.6369929, 3.4470274
3: -2.2489846, 3.6454558, -0.6862502, 0.9238169, -3.1728015, 4.3317060
4: -2.6095755, 3.6878364, -0.5532730, 1.1132669, -3.7228425, 4.2411094

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_B1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500071, upper bound: 1.0520318
time: 0.37 seconds

## Relational analysis of IS_A2_B1_B1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500071, upper bound: 1.0516579
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -1.1826556, 2.4220457, -0.1662051, 0.6552466, -1.8379022, 2.5882504
1: -1.6509128, 2.6472325, -0.3697689, 0.8258541, -2.4767668, 3.0170012
2: -1.6173604, 2.9671931, -0.2970747, 0.9300112, -2.5473716, 3.2642672
3: -2.1338735, 3.4227364, -0.6928098, 0.9316384, -3.0655117, 4.1155457
4: -2.4657319, 3.4712639, -0.5633935, 1.1270266, -3.5927584, 4.0346575

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0435574, upper bound: 1.0458322
time: 0.40 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A1_B2

### Relational analysis result of IS_A2_B1_B1_B1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0435574, upper bound: 1.0472275
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -1.2865010, 2.6105459, -0.1806883, 0.6643587, -1.9508598, 2.7912338
1: -1.7714832, 2.8234062, -0.3907865, 0.8332261, -2.6047094, 3.2141924
2: -1.7385111, 3.1969161, -0.3165183, 0.9454506, -2.6839616, 3.5134344
3: -2.2768168, 3.6857088, -0.7177029, 0.9569116, -3.2337284, 4.4034109
4: -2.6420763, 3.7331512, -0.5945891, 1.1518157, -3.7938919, 4.3277402

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B1_B2_A1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0482558, upper bound: 1.0488306
time: 0.35 seconds

## Relational analysis of IS_A2_B1_B1_B2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_B1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0484331, upper bound: 1.0499802
time: 0.40 seconds

## Relational analysis of IS_A2_B1_B1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0484331, upper bound: 1.0499801
time: 0.49 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -1.2995594, 2.6373627, -0.1905538, 0.6793638, -1.9789232, 2.8279166
1: -1.7885118, 2.8528142, -0.4057388, 0.8529248, -2.6414366, 3.2585528
2: -1.7535458, 3.2309113, -0.3302413, 0.9696531, -2.7231989, 3.5611522
3: -2.3000088, 3.7223716, -0.7378664, 0.9809170, -3.2809258, 4.4602380
4: -2.6641552, 3.7707973, -0.6153858, 1.1839089, -3.8480642, 4.3861828

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B1_B2_A1_B2_B1

### Relational analysis result of IS_A2_B1_B1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0482623, upper bound: 1.0499380
time: 0.41 seconds

## Relational analysis of IS_A2_B1_B1_B2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_B1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0484574, upper bound: 1.0501955
time: 0.44 seconds

## Relational analysis of IS_A2_B1_B1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0484574, upper bound: 1.0501955
time: 0.38 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 7.36 seconds
IS_A1_B1_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0517257, upper bound: 1.0509037
IS_A1_B1_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0517257, upper bound: 1.0509037
IS_A1_B1_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0517257, upper bound: 1.0509125
IS_A1_B1_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0517257, upper bound: 1.0509125
IS_A1_B1_A1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0529365, upper bound: 1.0505067
IS_A1_B1_A1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0529365, upper bound: 1.0505067
IS_A1_B1_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0529365, upper bound: 1.0506646
IS_A1_B1_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0529365, upper bound: 1.0506646
IS_A1_B1_A1_A2_B1_B1_B1, status: Status.VERIFIED, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0464128, upper bound: 1.0469908
IS_A1_B1_A1_A2_B1_B1_B2, status: Status.VERIFIED, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0463702, upper bound: 1.0461835
IS_A1_B1_A1_A2_B1_B2_B1, status: Status.VERIFIED, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0464128, upper bound: 1.0466616
IS_A1_B1_A1_A2_B1_B2_B2, status: Status.VERIFIED, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0463466, upper bound: 1.0465223
IS_A1_B1_A1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0499663, upper bound: 1.0499663
IS_A1_B1_A1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0499663, upper bound: 1.0499663
IS_A1_B1_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0499663, upper bound: 1.0501089
IS_A1_B1_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0499663, upper bound: 1.0501089
IS_A1_B1_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0509164, upper bound: 1.0505128
IS_A1_B1_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0509164, upper bound: 1.0505128
IS_A1_B1_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0511265, upper bound: 1.0505117
IS_A1_B1_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0511265, upper bound: 1.0505117
IS_A1_B1_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0499613, upper bound: 1.0502823
IS_A1_B1_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0499613, upper bound: 1.0502823
IS_A1_B1_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0501089, upper bound: 1.0502524
IS_A1_B1_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0501089, upper bound: 1.0502524
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0504873, upper bound: 1.0502489
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0504873, upper bound: 1.0502489
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0511128, upper bound: 1.0507046
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0509164, upper bound: 1.0507046
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0501823, upper bound: 1.0501999
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0501823, upper bound: 1.0501999
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0501824, upper bound: 1.0504029
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0501824, upper bound: 1.0504029
IS_A1_B2_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0516579, upper bound: 1.0500071
IS_A1_B2_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0516579, upper bound: 1.0500071
IS_A1_B2_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0516579, upper bound: 1.0500652
IS_A1_B2_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0516579, upper bound: 1.0500652
IS_A1_B2_A1_A1_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0458322, upper bound: 1.0435574
IS_A1_B2_A1_A1_B2_B1_A2, status: Status.VERIFIED, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0458322, upper bound: 1.0435574
IS_A1_B2_A1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0499801, upper bound: 1.0484593
IS_A1_B2_A1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0499801, upper bound: 1.0484593
IS_A1_B2_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0501955, upper bound: 1.0484574
IS_A1_B2_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0501955, upper bound: 1.0484574
IS_A1_B2_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0497719, upper bound: 1.0486099
IS_A1_B2_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0497720, upper bound: 1.0486730
IS_A1_B2_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0497720, upper bound: 1.0486099
IS_A1_B2_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0497720, upper bound: 1.0486730
IS_A1_B2_A2_A1_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0450647, upper bound: 1.0424943
IS_A1_B2_A2_A1_B1_B1_A2, status: Status.VERIFIED, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0430154, upper bound: 1.0422080
IS_A1_B2_A2_A1_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0450647, upper bound: 1.0424943
IS_A1_B2_A2_A1_B1_B2_A2, status: Status.VERIFIED, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0430154, upper bound: 1.0422080
IS_A1_B2_A2_A1_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0444293, upper bound: 1.0431251
IS_A1_B2_A2_A1_B2_B1_A2, status: Status.VERIFIED, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0444293, upper bound: 1.0431251
IS_A2_B1_B1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0500071, upper bound: 1.0516579
IS_A2_B1_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0500071, upper bound: 1.0516579
IS_A2_B1_B1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0500071, upper bound: 1.0520318
IS_A2_B1_B1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0500071, upper bound: 1.0516579
IS_A2_B1_B1_B1_A2_A1_B1, status: Status.VERIFIED, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0435574, upper bound: 1.0458322
IS_A2_B1_B1_B1_A2_A1_B2, status: Status.VERIFIED, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0435574, upper bound: 1.0472275
IS_A2_B1_B1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0484331, upper bound: 1.0499802
IS_A2_B1_B1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0484331, upper bound: 1.0499801
IS_A2_B1_B1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0484574, upper bound: 1.0501955
IS_A2_B1_B1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 7.36
Output dim: 0, lower bound: -1.0484574, upper bound: 1.0501955
IS_A2_B1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.36
Output dim: 0, lower bound: -1.0484331, upper bound: 1.0513621
IS_A2_B1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.36
Output dim: 0, lower bound: -1.0484331, upper bound: 1.0510623
IS_A2_B1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 7.36
Output dim: 0, lower bound: -1.0483126, upper bound: 1.0498558
IS_A2_B1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 7.36
Output dim: 0, lower bound: -1.0483126, upper bound: 1.0498831
IS_A2_B1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 7.36
Output dim: 0, lower bound: -1.0480907, upper bound: 1.0497811
IS_A2_B1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 7.36
Output dim: 0, lower bound: -1.0481636, upper bound: 1.0499567
IS_A2_B1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 7.36
Output dim: 0, lower bound: -1.0481636, upper bound: 1.0499567
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0500000, mid=0.0500000, abs_max=1.1883488893508911
rel_dist={0: [-1.0553818619849304, 1.0553818619849311]}

## Binary search (step 2) starts
Candidate diff: 0.0250000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0531974, upper bound: 1.0509638
time: 0.36 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
time: 0.35 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.86 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.86
Output dim: 0, lower bound: -1.0531974, upper bound: 1.0509638
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.86
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.2352601, 0.7538407, -0.3035725, 0.8847764, -1.1200366, 1.0574131
1: -0.4706453, 0.9488738, -0.5660125, 1.0933844, -1.5640295, 1.5148864
2: -0.3915833, 1.0723588, -0.4826685, 1.2412479, -1.6328310, 1.5550274
3: -0.8320177, 1.0878556, -0.9617165, 1.2755736, -2.1075912, 2.0495720
4: -0.7029035, 1.3031529, -0.8354526, 1.4994075, -2.2023110, 2.1386056

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
time: 0.37 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
time: 0.35 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -1.3271873, 2.7002649, -0.3000076, 0.8778248, -2.2050121, 3.0002723
1: -1.8266034, 2.9257355, -0.5613543, 1.0854805, -2.9120839, 3.4870896
2: -1.7866864, 3.3100519, -0.4781591, 1.2323816, -3.0190680, 3.7882106
3: -2.3538351, 3.8103127, -0.9558605, 1.2658806, -3.6197157, 4.7661729
4: -2.7129741, 3.8588223, -0.8289866, 1.4898851, -4.2028589, 4.6878090

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
time: 0.33 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
time: 0.33 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.28 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.28
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.28
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.28
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.28
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.2352601, 0.7538407, -0.2352601, 0.7538407, -0.9891008, 0.9891006
1: -0.4706453, 0.9488738, -0.4706453, 0.9488738, -1.4195192, 1.4195192
2: -0.3915833, 1.0723588, -0.3915833, 1.0723588, -1.4639422, 1.4639422
3: -0.8320177, 1.0878556, -0.8320177, 1.0878556, -1.9198732, 1.9198732
4: -0.7029035, 1.3031529, -0.7029035, 1.3031529, -2.0060563, 2.0060563

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0527347, upper bound: 1.0502364
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0526130, upper bound: 1.0508594
time: 0.38 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.2352601, 0.7538407, -1.3271873, 2.7002649, -2.9355247, 2.0810280
1: -0.4706453, 0.9488738, -1.8266034, 2.9257355, -3.3963809, 2.7754772
2: -0.3915833, 1.0723588, -1.7866864, 3.3100519, -3.7016354, 2.8590453
3: -0.8320177, 1.0878556, -2.3538351, 3.8103127, -4.6423302, 3.4416907
4: -0.7029035, 1.3031529, -2.7129741, 3.8588223, -4.5617256, 4.0161266

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0527633, upper bound: 1.0508775
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0517363, upper bound: 1.0507646
time: 0.34 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -1.3271873, 2.7002649, -0.2352601, 0.7538407, -2.0810280, 2.9355249
1: -1.8266034, 2.9257355, -0.4706453, 0.9488738, -2.7754772, 3.3963809
2: -1.7866864, 3.3100519, -0.3915833, 1.0723588, -2.8590453, 3.7016351
3: -2.3538351, 3.8103127, -0.8320177, 1.0878556, -3.4416907, 4.6423302
4: -2.7129741, 3.8588223, -0.7029035, 1.3031529, -4.0161266, 4.5617256

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507166, upper bound: 1.0507511
time: 0.34 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507030, upper bound: 1.0507030
time: 0.39 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -1.3271873, 2.7002649, -1.3271873, 2.7002649, -4.0274525, 4.0274525
1: -1.8266034, 2.9257355, -1.8266034, 2.9257355, -4.7523389, 4.7523384
2: -1.7866864, 3.3100519, -1.7866864, 3.3100519, -5.0967383, 5.0967379
3: -2.3538351, 3.8103127, -2.3538351, 3.8103127, -6.1641479, 6.1641479
4: -2.7129741, 3.8588223, -2.7129741, 3.8588223, -6.5717964, 6.5717964

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0483393, upper bound: 1.0488063
time: 0.36 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506395, upper bound: 1.0506395
time: 0.43 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.45 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.45
Output dim: 0, lower bound: -1.0527347, upper bound: 1.0502364
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.45
Output dim: 0, lower bound: -1.0526130, upper bound: 1.0508594
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.45
Output dim: 0, lower bound: -1.0527633, upper bound: 1.0508775
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.45
Output dim: 0, lower bound: -1.0517363, upper bound: 1.0507646
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 2.45
Output dim: 0, lower bound: -1.0507166, upper bound: 1.0507511
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 2.45
Output dim: 0, lower bound: -1.0507030, upper bound: 1.0507030
IS_A2_B2_B1, status: Status.VERIFIED, split count: 3, time: 2.45
Output dim: 0, lower bound: -1.0483393, upper bound: 1.0488063
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 2.45
Output dim: 0, lower bound: -1.0506395, upper bound: 1.0506395

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.2013956, 0.7023641, -0.2201980, 0.7295994, -0.9309949, 0.9225621
1: -0.4206367, 0.8831170, -0.4483676, 0.9182405, -1.3388772, 1.3314846
2: -0.3466128, 1.0003821, -0.3716945, 1.0391469, -1.3857597, 1.3720765
3: -0.7602279, 1.0080743, -0.8000128, 1.0511825, -1.8114104, 1.8080871
4: -0.6370696, 1.2147777, -0.6739963, 1.2631135, -1.9001831, 1.8887740

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535536, upper bound: 1.0528018
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0532310, upper bound: 1.0520455
time: 0.41 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.2363607, 0.7416377, -0.2207971, 0.7253933, -0.9617540, 0.9624348
1: -0.4722191, 0.9222932, -0.4493197, 0.9112105, -1.3834295, 1.3716129
2: -0.3964722, 1.0577438, -0.3732101, 1.0342234, -1.4306957, 1.4309539
3: -0.8244337, 1.0810699, -0.8019392, 1.0477722, -1.8722059, 1.8830092
4: -0.7141775, 1.3009543, -0.6769375, 1.2616144, -1.9757919, 1.9778919

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539857, upper bound: 1.0540170
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539857, upper bound: 1.0540170
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.2121267, 0.7201707, -1.3271873, 2.7002649, -2.9123917, 2.0473580
1: -0.4377390, 0.9047500, -1.8266034, 2.9257355, -3.3634744, 2.7313535
2: -0.3605663, 1.0264064, -1.7866864, 3.3100519, -3.6706183, 2.8130927
3: -0.7864621, 1.0343318, -2.3538351, 3.8103127, -4.5967746, 3.3881669
4: -0.6582627, 1.2474420, -2.7129741, 3.8588223, -4.5170841, 3.9604161

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0516959, upper bound: 1.0495409
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0523143, upper bound: 1.0507842
time: 0.48 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.2998355, 0.9004637, -1.2347200, 2.5075514, -2.8073864, 2.1351833
1: -0.5571232, 1.1422836, -1.7044203, 2.7145898, -3.2717130, 2.8467038
2: -0.4746163, 1.2865998, -1.6690869, 3.0891764, -3.5637927, 2.9556866
3: -0.9696201, 1.2997241, -2.2076201, 3.5399663, -4.5095859, 3.5073435
4: -0.8469371, 1.4981039, -2.5458355, 3.6071315, -4.4540677, 4.0439391

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500034, upper bound: 1.0486979
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0514137, upper bound: 1.0506707
time: 0.44 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -1.3271873, 2.7002649, -0.2121267, 0.7201707, -2.0473580, 2.9123917
1: -1.8266034, 2.9257355, -0.4377390, 0.9047500, -2.7313535, 3.3634744
2: -1.7866864, 3.3100519, -0.3605663, 1.0264064, -2.8130927, 3.6706183
3: -2.3538351, 3.8103127, -0.7864621, 1.0343318, -3.3881669, 4.5967746
4: -2.7129741, 3.8588223, -0.6582627, 1.2474420, -3.9604161, 4.5170846

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0495409, upper bound: 1.0516959
time: 0.42 seconds

## Relational analysis of IS_A2_B1_B1_B2

### Relational analysis result of IS_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0495409, upper bound: 1.0523143
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -1.2347200, 2.5075514, -0.2998355, 0.9004637, -2.1351833, 2.8073866
1: -1.7044203, 2.7145898, -0.5571232, 1.1422836, -2.8467038, 3.2717130
2: -1.6690869, 3.0891764, -0.4746163, 1.2865998, -2.9556866, 3.5637927
3: -2.2076201, 3.5399663, -0.9696201, 1.2997241, -3.5073435, 4.5095859
4: -2.5458355, 3.6071315, -0.8469371, 1.4981039, -4.0439386, 4.4540677

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0486979, upper bound: 1.0500034
time: 0.37 seconds

## Relational analysis of IS_A2_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0486979, upper bound: 1.0514137
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -1.3012817, 2.6305966, -1.3654406, 2.6592798, -3.9605615, 3.9960370
1: -1.7898400, 2.8466678, -1.8352041, 2.8745258, -4.6643658, 4.6818719
2: -1.7561550, 3.2289431, -1.7965705, 3.2761538, -5.0323086, 5.0255122
3: -2.3022947, 3.7166531, -2.3616958, 3.7730355, -6.0753298, 6.0783491
4: -2.6689839, 3.7697055, -2.7261250, 3.8257360, -6.4947195, 6.4958305

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0488063, upper bound: 1.0483393
time: 0.35 seconds

## Relational analysis of IS_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0488063, upper bound: 1.0506395
time: 0.37 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.35 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -1.0535536, upper bound: 1.0528018
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -1.0532310, upper bound: 1.0520455
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -1.0539857, upper bound: 1.0540170
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -1.0539857, upper bound: 1.0540170
IS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -1.0516959, upper bound: 1.0495409
IS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -1.0523143, upper bound: 1.0507842
IS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -1.0500034, upper bound: 1.0486979
IS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -1.0514137, upper bound: 1.0506707
IS_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -1.0495409, upper bound: 1.0516959
IS_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -1.0495409, upper bound: 1.0523143
IS_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -1.0486979, upper bound: 1.0500034
IS_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -1.0486979, upper bound: 1.0514137
IS_A2_B2_B2_A1, status: Status.VERIFIED, split count: 4, time: 2.35
Output dim: 0, lower bound: -1.0488063, upper bound: 1.0483393
IS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -1.0488063, upper bound: 1.0506395

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1896522, 0.6858314, -0.1872458, 0.6823201, -0.8719723, 0.8730772
1: -0.4031372, 0.8651893, -0.3991195, 0.8628459, -1.2659831, 1.2643087
2: -0.3294382, 0.9751638, -0.3237809, 0.9684508, -1.2978890, 1.2989447
3: -0.7384253, 0.9815660, -0.7379917, 0.9756166, -1.7140419, 1.7195576
4: -0.6110716, 1.1812500, -0.6011894, 1.1687107, -1.7797823, 1.7824394

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513698, upper bound: 1.0503224
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501064, upper bound: 1.0490627
time: 0.38 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1949764, 0.6936327, -0.1968904, 0.6979628, -0.8929392, 0.8905231
1: -0.4112571, 0.8747689, -0.4142706, 0.8859708, -1.2972280, 1.2890396
2: -0.3369835, 0.9875940, -0.3372641, 0.9931101, -1.3300936, 1.3248581
3: -0.7484866, 0.9943962, -0.7572864, 1.0011578, -1.7496443, 1.7516826
4: -0.6223364, 1.1972525, -0.6216018, 1.2001703, -1.8225067, 1.8188543

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0516296, upper bound: 1.0504071
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501464, upper bound: 1.0501089
time: 0.37 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.2363607, 0.7416377, -0.2013956, 0.7023641, -0.9387248, 0.9430333
1: -0.4722191, 0.9222932, -0.4206367, 0.8831170, -1.3553361, 1.3429298
2: -0.3964722, 1.0577438, -0.3466128, 1.0003821, -1.3968543, 1.4043566
3: -0.8244337, 1.0810699, -0.7602279, 1.0080743, -1.8325080, 1.8412979
4: -0.7141775, 1.3009543, -0.6370696, 1.2147777, -1.9289553, 1.9380239

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0521540, upper bound: 1.0525930
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0519933, upper bound: 1.0519933
time: 0.42 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.2363607, 0.7416377, -0.2363607, 0.7416377, -0.9779984, 0.9779984
1: -0.4722191, 0.9222932, -0.4722191, 0.9222932, -1.3945123, 1.3945123
2: -0.3964722, 1.0577438, -0.3964722, 1.0577438, -1.4542160, 1.4542160
3: -0.8244337, 1.0810699, -0.8244337, 1.0810699, -1.9055036, 1.9055036
4: -0.7141775, 1.3009543, -0.7141775, 1.3009543, -2.0151320, 2.0151320

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0526034, upper bound: 1.0521540
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0519933, upper bound: 1.0519933
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.1851257, 0.6783313, -1.3166026, 2.6747036, -2.8598294, 1.9949338
1: -0.3969364, 0.8526834, -1.8116517, 2.8957775, -3.2927136, 2.6643350
2: -0.3242711, 0.9644121, -1.7737854, 3.2782125, -3.6024837, 2.7381973
3: -0.7251614, 0.9704387, -2.3325071, 3.7746639, -4.4998255, 3.3029459
4: -0.6038694, 1.1718525, -2.6938512, 3.8235836, -4.4274530, 3.8657036

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_A1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507517, upper bound: 1.0487464
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508334, upper bound: 1.0486791
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.2122585, 0.7096910, -1.3012817, 2.6305966, -2.8428552, 2.0109727
1: -0.4381096, 0.8818882, -1.7898400, 2.8466678, -3.2847772, 2.6717281
2: -0.3631817, 1.0126708, -1.7561550, 3.2289431, -3.5921240, 2.7688258
3: -0.7779223, 1.0290604, -2.3022947, 3.7166531, -4.4945755, 3.3313551
4: -0.6656082, 1.2431200, -2.6689839, 3.7697055, -4.4353137, 3.9121039

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_A2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507175, upper bound: 1.0484881
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507175, upper bound: 1.0507842
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.2699254, 0.8529468, -1.2228200, 2.4776444, -2.7475698, 2.0757666
1: -0.5107998, 1.0817231, -1.6876335, 2.6793957, -3.1901951, 2.7693563
2: -0.4349182, 1.2233939, -1.6549249, 3.0518506, -3.4867687, 2.8783188
3: -0.9060124, 1.2251596, -2.1833670, 3.4987981, -4.4048104, 3.4085267
4: -0.7885253, 1.4212955, -2.5248837, 3.5662956, -4.3548212, 3.9461792

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_A1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0494752, upper bound: 1.0466427
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0494752, upper bound: 1.0480971
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.3204554, 0.9300180, -1.2133427, 2.4497728, -2.7702281, 2.1433606
1: -0.5845242, 1.1713222, -1.6736221, 2.6495252, -3.2340493, 2.8449445
2: -0.5056426, 1.3326001, -1.6436391, 3.0219576, -3.5276003, 2.9762392
3: -1.0023541, 1.3492649, -2.1643989, 3.4622431, -4.4645972, 3.5136635
4: -0.8973903, 1.5542920, -2.5094881, 3.5324621, -4.4298525, 4.0637798

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0497983, upper bound: 1.0483607
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2

### Relational analysis result of IS_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0497983, upper bound: 1.0483607
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -1.3166026, 2.6747036, -0.1851257, 0.6783313, -1.9949338, 2.8598289
1: -1.8116517, 2.8957775, -0.3969364, 0.8526834, -2.6643350, 3.2927134
2: -1.7737854, 3.2782125, -0.3242711, 0.9644121, -2.7381973, 3.6024837
3: -2.3325071, 3.7746639, -0.7251614, 0.9704387, -3.3029459, 4.4998255
4: -2.6938512, 3.8235836, -0.6038694, 1.1718525, -3.8657036, 4.4274526

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0487464, upper bound: 1.0507517
time: 0.37 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0486791, upper bound: 1.0508334
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -1.3012817, 2.6305966, -0.2122585, 0.7096910, -2.0109727, 2.8428552
1: -1.7898400, 2.8466678, -0.4381096, 0.8818882, -2.6717281, 3.2847772
2: -1.7561550, 3.2289431, -0.3631817, 1.0126708, -2.7688258, 3.5921245
3: -2.3022947, 3.7166531, -0.7779223, 1.0290604, -3.3313551, 4.4945755
4: -2.6689839, 3.7697055, -0.6656082, 1.2431200, -3.9121039, 4.4353137

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_B1_B2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0484881, upper bound: 1.0507175
time: 0.42 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0484881, upper bound: 1.0523143
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -1.2228200, 2.4776444, -0.2699254, 0.8529468, -2.0757666, 2.7475698
1: -1.6876335, 2.6793957, -0.5107998, 1.0817231, -2.7693563, 3.1901951
2: -1.6549249, 3.0518506, -0.4349182, 1.2233939, -2.8783188, 3.4867687
3: -2.1833670, 3.4987981, -0.9060124, 1.2251596, -3.4085267, 4.4048104
4: -2.5248837, 3.5662956, -0.7885253, 1.4212955, -3.9461792, 4.3548207

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B2_B1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0466427, upper bound: 1.0494752
time: 0.40 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0480971, upper bound: 1.0496100
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -1.2133427, 2.4497728, -0.3204554, 0.9300180, -2.1433606, 2.7702279
1: -1.6736221, 2.6495252, -0.5845242, 1.1713222, -2.8449445, 3.2340493
2: -1.6436391, 3.0219576, -0.5056426, 1.3326001, -2.9762392, 3.5276003
3: -2.1643989, 3.4622431, -1.0023541, 1.3492649, -3.5136638, 4.4645972
4: -2.5094881, 3.5324621, -0.8973903, 1.5542920, -4.0637798, 4.4298525

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_B2_B2_A1

### Relational analysis result of IS_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0483607, upper bound: 1.0497983
time: 0.40 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2

### Relational analysis result of IS_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0483607, upper bound: 1.0514137
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -1.3654406, 2.6592798, -1.3654406, 2.6592798, -4.0247202, 4.0247202
1: -1.8352041, 2.8745258, -1.8352041, 2.8745258, -4.7097301, 4.7097301
2: -1.7965705, 3.2761538, -1.7965705, 3.2761538, -5.0727239, 5.0727239
3: -2.3616958, 3.7730355, -2.3616958, 3.7730355, -6.1347303, 6.1347303
4: -2.7261250, 3.8257360, -2.7261250, 3.8257360, -6.5518599, 6.5518599

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_B2_A2_B1

### Relational analysis result of IS_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0474032, upper bound: 1.0506259
time: 0.40 seconds

## Relational analysis of IS_A2_B2_B2_A2_B2

### Relational analysis result of IS_A2_B2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0472169, upper bound: 1.0472169
time: 0.36 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.51 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.51
Output dim: 0, lower bound: -1.0513698, upper bound: 1.0503224
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.51
Output dim: 0, lower bound: -1.0501064, upper bound: 1.0490627
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.51
Output dim: 0, lower bound: -1.0516296, upper bound: 1.0504071
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.51
Output dim: 0, lower bound: -1.0501464, upper bound: 1.0501089
IS_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.51
Output dim: 0, lower bound: -1.0521540, upper bound: 1.0525930
IS_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.51
Output dim: 0, lower bound: -1.0519933, upper bound: 1.0519933
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.51
Output dim: 0, lower bound: -1.0526034, upper bound: 1.0521540
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.51
Output dim: 0, lower bound: -1.0519933, upper bound: 1.0519933
IS_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.51
Output dim: 0, lower bound: -1.0507517, upper bound: 1.0487464
IS_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.51
Output dim: 0, lower bound: -1.0508334, upper bound: 1.0486791
IS_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.51
Output dim: 0, lower bound: -1.0507175, upper bound: 1.0484881
IS_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.51
Output dim: 0, lower bound: -1.0507175, upper bound: 1.0507842
IS_A1_B2_A2_A1_B1, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 0, lower bound: -1.0494752, upper bound: 1.0466427
IS_A1_B2_A2_A1_B2, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 0, lower bound: -1.0494752, upper bound: 1.0480971
IS_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.51
Output dim: 0, lower bound: -1.0497983, upper bound: 1.0483607
IS_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.51
Output dim: 0, lower bound: -1.0497983, upper bound: 1.0483607
IS_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.51
Output dim: 0, lower bound: -1.0487464, upper bound: 1.0507517
IS_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.51
Output dim: 0, lower bound: -1.0486791, upper bound: 1.0508334
IS_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.51
Output dim: 0, lower bound: -1.0484881, upper bound: 1.0507175
IS_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.51
Output dim: 0, lower bound: -1.0484881, upper bound: 1.0523143
IS_A2_B1_B2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 0, lower bound: -1.0466427, upper bound: 1.0494752
IS_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.51
Output dim: 0, lower bound: -1.0480971, upper bound: 1.0496100
IS_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.51
Output dim: 0, lower bound: -1.0483607, upper bound: 1.0497983
IS_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.51
Output dim: 0, lower bound: -1.0483607, upper bound: 1.0514137
IS_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.51
Output dim: 0, lower bound: -1.0474032, upper bound: 1.0506259
IS_A2_B2_B2_A2_B2, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 0, lower bound: -1.0472169, upper bound: 1.0472169

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1730340, 0.6613784, -0.1872458, 0.6823201, -0.8553541, 0.8486242
1: -0.3788329, 0.8342032, -0.3991195, 0.8628459, -1.2416788, 1.2333226
2: -0.3065768, 0.9387357, -0.3237809, 0.9684508, -1.2750275, 1.2625165
3: -0.7025846, 0.9433535, -0.7379917, 0.9756166, -1.6782012, 1.6813452
4: -0.5771155, 1.1376231, -0.6011894, 1.1687107, -1.7458262, 1.7388124

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513698, upper bound: 1.0499766
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501064, upper bound: 1.0488845
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501064, upper bound: 1.0490627
time: 0.38 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.2542607, 0.8079799, -0.1567561, 0.6325021, -0.8867628, 0.9647360
1: -0.4890153, 1.0335444, -0.3519635, 0.8029545, -1.2919698, 1.3855079
2: -0.4136510, 1.1714401, -0.2765362, 0.9036011, -1.3172522, 1.4479764
3: -0.8771543, 1.1680876, -0.6831658, 0.8957011, -1.7728554, 1.8512535
4: -0.7573829, 1.3704841, -0.5313109, 1.0837650, -1.8411479, 1.9017949

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501064, upper bound: 1.0488845
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501064, upper bound: 1.0490627
time: 0.44 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1786847, 0.6695686, -0.1968904, 0.6979628, -0.8766475, 0.8664590
1: -0.3875105, 0.8442428, -0.4142706, 0.8859708, -1.2734814, 1.2585135
2: -0.3145856, 0.9516206, -0.3372641, 0.9931101, -1.3076956, 1.2888846
3: -0.7133602, 0.9566646, -0.7572864, 1.0011578, -1.7145180, 1.7139510
4: -0.5890352, 1.1542461, -0.6216018, 1.2001703, -1.7892054, 1.7758479

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0516123, upper bound: 1.0502515
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0516123, upper bound: 1.0504071
time: 0.42 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.2604521, 0.8180948, -0.1646138, 0.6458658, -0.9063179, 0.9827087
1: -0.4973333, 1.0453314, -0.3630166, 0.8225024, -1.3198357, 1.4083480
2: -0.4221133, 1.1859961, -0.2884538, 0.9240967, -1.3462100, 1.4744499
3: -0.8889325, 1.1832230, -0.6987863, 0.9181318, -1.8070643, 1.8820093
4: -0.7701943, 1.3884196, -0.5511286, 1.1082954, -1.8784897, 1.9395483

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499663, upper bound: 1.0499663
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499663, upper bound: 1.0501089
time: 0.42 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.2235301, 0.7230793, -0.1708286, 0.6587178, -0.8822479, 0.8939079
1: -0.4532241, 0.9027213, -0.3748004, 0.8337236, -1.2869477, 1.2775218
2: -0.3776526, 1.0300139, -0.3015155, 0.9337045, -1.3113571, 1.3315294
3: -0.8006500, 1.0514371, -0.7031220, 0.9388585, -1.7395085, 1.7545592
4: -0.6856011, 1.2635684, -0.5683438, 1.1266637, -1.8122648, 1.8319122

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513650, upper bound: 1.0506779
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0521540, upper bound: 1.0526937
time: 0.44 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.2300024, 0.7326542, -0.1796174, 0.6729426, -0.9029450, 0.9122715
1: -0.4627191, 0.9136744, -0.3886608, 0.8546216, -1.3173406, 1.3023351
2: -0.3868110, 1.0446260, -0.3137678, 0.9577398, -1.3445508, 1.3583938
3: -0.8125668, 1.0666299, -0.7210382, 0.9618257, -1.7743926, 1.7876681
4: -0.6995184, 1.2829386, -0.5868647, 1.1561475, -1.8556659, 1.8698033

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507258, upper bound: 1.0502713
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501089, upper bound: 1.0501464
time: 0.48 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.2045429, 0.6960640, -0.2235301, 0.7230793, -0.9276223, 0.9195942
1: -0.4246404, 0.8731498, -0.4532241, 0.9027213, -1.3273618, 1.3263738
2: -0.3495150, 0.9898366, -0.3776526, 1.0300139, -1.3795289, 1.3674893
3: -0.7641593, 1.0075936, -0.8006500, 1.0514371, -1.8155963, 1.8082436
4: -0.6426289, 1.2076913, -0.6856011, 1.2635684, -1.9061973, 1.8932924

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0519933, upper bound: 1.0519933
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0519933, upper bound: 1.0519933
time: 0.38 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.2148956, 0.7115479, -0.2300024, 0.7326542, -0.9475498, 0.9415503
1: -0.4401824, 0.8935100, -0.4627191, 0.9136744, -1.3538568, 1.3562291
2: -0.3639436, 1.0145798, -0.3868110, 1.0446260, -1.4085696, 1.4013908
3: -0.7847203, 1.0327761, -0.8125668, 1.0666299, -1.8513502, 1.8453429
4: -0.6645471, 1.2404692, -0.6995184, 1.2829386, -1.9474857, 1.9399877

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504844, upper bound: 1.0510743
time: 0.49 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502161, upper bound: 1.0503624
time: 0.44 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1730340, 0.6613784, -1.2549818, 2.5635638, -2.7365978, 1.9163601
1: -0.3788329, 0.8342032, -1.7373600, 2.7744865, -3.1533191, 2.5715632
2: -0.3065768, 0.9387357, -1.7065775, 3.1345379, -3.4411147, 2.6453133
3: -0.7025846, 0.9433535, -2.2345841, 3.6214995, -4.3240843, 3.1779375
4: -0.5771155, 1.1376231, -2.5968800, 3.6642756, -4.2413912, 3.7345030

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507481, upper bound: 1.0473869
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507517, upper bound: 1.0487464
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507517, upper bound: 1.0487464
time: 0.44 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1786847, 0.6695686, -1.2804995, 2.6228871, -2.8015718, 1.9500680
1: -0.3875105, 0.8442428, -1.7674365, 2.8425279, -3.2300384, 2.6116793
2: -0.3145856, 0.9516206, -1.7332027, 3.2041645, -3.5187502, 2.6848233
3: -0.7133602, 0.9566646, -2.2757225, 3.7009442, -4.4143033, 3.2323871
4: -0.5890352, 1.1542461, -2.6366940, 3.7360914, -4.3251266, 3.7909400

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508334, upper bound: 1.0483984
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0457103, upper bound: 1.0433378
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.2122585, 0.7096910, -1.3023098, 2.6408219, -2.8530805, 2.0120008
1: -0.4381096, 0.8818882, -1.7915673, 2.8563752, -3.2944846, 2.6734555
2: -0.3631817, 1.0126708, -1.7563353, 3.2358289, -3.5990105, 2.7690060
3: -0.7779223, 1.0290604, -2.3039412, 3.7273993, -4.5053215, 3.3330016
4: -0.6656082, 1.2431200, -2.6680644, 3.7765968, -4.4422050, 3.9111843

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0453804, upper bound: 1.0435455
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479240, upper bound: 1.0480780
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479240, upper bound: 1.0481880
time: 0.46 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.2122585, 0.7096910, -1.3654406, 2.6592798, -2.8715384, 2.0751317
1: -0.4381096, 0.8818882, -1.8352041, 2.8745258, -3.3126354, 2.7170923
2: -0.3631817, 1.0126708, -1.7965705, 3.2761538, -3.6393356, 2.8092413
3: -0.7779223, 1.0290604, -2.3616958, 3.7730355, -4.5509577, 3.3907561
4: -0.6656082, 1.2431200, -2.7261250, 3.8257360, -4.4913445, 3.9692450

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0453804, upper bound: 1.0435455
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479240, upper bound: 1.0489614
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0497963, upper bound: 1.0486588
time: 0.48 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.3204554, 0.9300180, -1.2105802, 2.4459045, -2.7663598, 2.1405983
1: -0.5845242, 1.1713222, -1.6696608, 2.6414123, -3.2259362, 2.8409829
2: -0.5056426, 1.3326001, -1.6397901, 3.0118001, -3.5174427, 2.9723902
3: -1.0023541, 1.3492649, -2.1572886, 3.4545929, -4.4569468, 3.5065532
4: -0.8973903, 1.5542920, -2.5024679, 3.5223804, -4.4197702, 4.0567598

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0447033, upper bound: 1.0434794
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480903, upper bound: 1.0480393
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0491765, upper bound: 1.0480736
time: 0.45 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.3204554, 0.9300180, -1.2859970, 2.4919481, -2.8124032, 2.2160151
1: -0.5845242, 1.1713222, -1.7275903, 2.6922684, -3.2767925, 2.8989124
2: -0.5056426, 1.3326001, -1.6930435, 3.0845804, -3.5902231, 3.0256433
3: -1.0023541, 1.3492649, -2.2335849, 3.5380590, -4.5404129, 3.5828497
4: -0.8973903, 1.5542920, -2.5790989, 3.6047988, -4.5021892, 4.1333909

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0448802, upper bound: 1.0434794
time: 0.44 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480903, upper bound: 1.0488486
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0491765, upper bound: 1.0486524
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -1.2549818, 2.5635638, -0.1730340, 0.6613784, -1.9163601, 2.7365978
1: -1.7373600, 2.7744865, -0.3788329, 0.8342032, -2.5715632, 3.1533191
2: -1.7065775, 3.1345379, -0.3065768, 0.9387357, -2.6453133, 3.4411144
3: -2.2345841, 3.6214995, -0.7025846, 0.9433535, -3.1779375, 4.3240843
4: -2.5968800, 3.6642756, -0.5771155, 1.1376231, -3.7345030, 4.2413912

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0473869, upper bound: 1.0507481
time: 0.37 seconds

## Relational analysis of IS_A2_B1_B1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_B1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0487464, upper bound: 1.0507517
time: 0.38 seconds

## Relational analysis of IS_A2_B1_B1_B1_A1_A2

### Relational analysis result of IS_A2_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0487464, upper bound: 1.0507517
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -1.2804995, 2.6228871, -0.1786847, 0.6695686, -1.9500680, 2.8015718
1: -1.7674365, 2.8425279, -0.3875105, 0.8442428, -2.6116793, 3.2300384
2: -1.7332027, 3.2041645, -0.3145856, 0.9516206, -2.6848233, 3.5187497
3: -2.2757225, 3.7009442, -0.7133602, 0.9566646, -3.2323871, 4.4143038
4: -2.6366940, 3.7360914, -0.5890352, 1.1542461, -3.7909400, 4.3251262

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0483984, upper bound: 1.0508334
time: 0.40 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A2

### Relational analysis result of IS_A2_B1_B1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0433378, upper bound: 1.0457103
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -1.3023098, 2.6408219, -0.2122585, 0.7096910, -2.0120008, 2.8530805
1: -1.7915673, 2.8563752, -0.4381096, 0.8818882, -2.6734555, 3.2944846
2: -1.7563353, 3.2358289, -0.3631817, 1.0126708, -2.7690060, 3.5990105
3: -2.3039412, 3.7273993, -0.7779223, 1.0290604, -3.3330016, 4.5053215
4: -2.6680644, 3.7765968, -0.6656082, 1.2431200, -3.9111843, 4.4422050

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0435455, upper bound: 1.0453804
time: 0.38 seconds

## Relational analysis of IS_A2_B1_B1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_B1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0484533, upper bound: 1.0507175
time: 0.42 seconds

## Relational analysis of IS_A2_B1_B1_B2_A1_A2

### Relational analysis result of IS_A2_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0484533, upper bound: 1.0507175
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -1.3654406, 2.6592798, -0.2122585, 0.7096910, -2.0751317, 2.8715384
1: -1.8352041, 2.8745258, -0.4381096, 0.8818882, -2.7170923, 3.3126349
2: -1.7965705, 3.2761538, -0.3631817, 1.0126708, -2.8092413, 3.6393352
3: -2.3616958, 3.7730355, -0.7779223, 1.0290604, -3.3907561, 4.5509577
4: -2.7261250, 3.8257360, -0.6656082, 1.2431200, -3.9692450, 4.4913440

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0435455, upper bound: 1.0453804
time: 0.37 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_B1_B2_A2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0484533, upper bound: 1.0507175
time: 0.40 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0484533, upper bound: 1.0507175
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -1.1878020, 2.4265501, -0.2627929, 0.8424390, -2.0302410, 2.6893427
1: -1.6438856, 2.6266255, -0.5004367, 1.0710534, -2.7149391, 3.1270618
2: -1.6144052, 2.9784088, -0.4241708, 1.2085010, -2.8229060, 3.4025793
3: -2.1277685, 3.4255252, -0.8929762, 1.2087853, -3.3365538, 4.3185015
4: -2.4674721, 3.4791443, -0.7722017, 1.4005836, -3.8680553, 4.2513461

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B2_B1_A2_A1

### Relational analysis result of IS_A2_B1_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0478559, upper bound: 1.0494273
time: 0.41 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2_A2

### Relational analysis result of IS_A2_B1_B2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0431251, upper bound: 1.0444293
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -1.2105802, 2.4459045, -0.3204554, 0.9300180, -2.1405983, 2.7663598
1: -1.6696608, 2.6414123, -0.5845242, 1.1713222, -2.8409829, 3.2259364
2: -1.6397901, 3.0118001, -0.5056426, 1.3326001, -2.9723902, 3.5174427
3: -2.1572886, 3.4545929, -1.0023541, 1.3492649, -3.5065532, 4.4569464
4: -2.5024679, 3.5223804, -0.8973903, 1.5542920, -4.0567598, 4.4197707

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B2_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0433971, upper bound: 1.0448773
time: 0.36 seconds

## Relational analysis of IS_A2_B1_B2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_B2_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0481018, upper bound: 1.0497983
time: 0.37 seconds

## Relational analysis of IS_A2_B1_B2_B2_A1_A2

### Relational analysis result of IS_A2_B1_B2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0481018, upper bound: 1.0493216
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -1.2859970, 2.4919481, -0.3204554, 0.9300180, -2.2160151, 2.8124032
1: -1.7275903, 2.6922684, -0.5845242, 1.1713222, -2.8989124, 3.2767925
2: -1.6930435, 3.0845804, -0.5056426, 1.3326001, -3.0256433, 3.5902231
3: -2.2335849, 3.5380590, -1.0023541, 1.3492649, -3.5828497, 4.5404129
4: -2.5790989, 3.6047988, -0.8973903, 1.5542920, -4.1333904, 4.5021892

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B2_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0433971, upper bound: 1.0447033
time: 0.51 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_B2_B2_A2_A1

### Relational analysis result of IS_A2_B1_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0481018, upper bound: 1.0493216
time: 0.38 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2_A2

### Relational analysis result of IS_A2_B1_B2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0481018, upper bound: 1.0493216
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -1.3654406, 2.6592798, -1.3498325, 2.6225345, -3.9879746, 4.0091124
1: -1.8352041, 2.8745258, -1.8129959, 2.8317785, -4.6669827, 4.6875219
2: -1.7965705, 3.2761538, -1.7776842, 3.2286239, -5.0251942, 5.0538373
3: -2.3616958, 3.7730355, -2.3294129, 3.7206798, -6.0823750, 6.1024485
4: -2.7261250, 3.8257360, -2.6985631, 3.7726209, -6.4987450, 6.5242987

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0425972, upper bound: 1.0428639
time: 0.43 seconds

## Relational analysis of IS_A2_B2_B2_A2_B1_B2

### Relational analysis result of IS_A2_B2_B2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0257711, upper bound: 1.0272760
time: 0.43 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.62 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 0, lower bound: -1.0501064, upper bound: 1.0488845
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 0, lower bound: -1.0501064, upper bound: 1.0490627
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 0, lower bound: -1.0501064, upper bound: 1.0488845
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 0, lower bound: -1.0501064, upper bound: 1.0490627
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 0, lower bound: -1.0516123, upper bound: 1.0502515
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 0, lower bound: -1.0516123, upper bound: 1.0504071
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 0, lower bound: -1.0499663, upper bound: 1.0499663
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 0, lower bound: -1.0499663, upper bound: 1.0501089
IS_A1_B1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 0, lower bound: -1.0513650, upper bound: 1.0506779
IS_A1_B1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 0, lower bound: -1.0521540, upper bound: 1.0526937
IS_A1_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 0, lower bound: -1.0507258, upper bound: 1.0502713
IS_A1_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 0, lower bound: -1.0501089, upper bound: 1.0501464
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 0, lower bound: -1.0519933, upper bound: 1.0519933
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 0, lower bound: -1.0519933, upper bound: 1.0519933
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 0, lower bound: -1.0504844, upper bound: 1.0510743
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 0, lower bound: -1.0502161, upper bound: 1.0503624
IS_A1_B2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 0, lower bound: -1.0507517, upper bound: 1.0487464
IS_A1_B2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 0, lower bound: -1.0507517, upper bound: 1.0487464
IS_A1_B2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 0, lower bound: -1.0508334, upper bound: 1.0483984
IS_A1_B2_A1_A1_B2_B2, status: Status.VERIFIED, split count: 6, time: 2.62
Output dim: 0, lower bound: -1.0457103, upper bound: 1.0433378
IS_A1_B2_A1_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 2.62
Output dim: 0, lower bound: -1.0479240, upper bound: 1.0480780
IS_A1_B2_A1_A2_B1_A2, status: Status.VERIFIED, split count: 6, time: 2.62
Output dim: 0, lower bound: -1.0479240, upper bound: 1.0481880
IS_A1_B2_A1_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 2.62
Output dim: 0, lower bound: -1.0479240, upper bound: 1.0489614
IS_A1_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 0, lower bound: -1.0497963, upper bound: 1.0486588
IS_A1_B2_A2_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 2.62
Output dim: 0, lower bound: -1.0480903, upper bound: 1.0480393
IS_A1_B2_A2_A2_B1_A2, status: Status.VERIFIED, split count: 6, time: 2.62
Output dim: 0, lower bound: -1.0491765, upper bound: 1.0480736
IS_A1_B2_A2_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 2.62
Output dim: 0, lower bound: -1.0480903, upper bound: 1.0488486
IS_A1_B2_A2_A2_B2_A2, status: Status.VERIFIED, split count: 6, time: 2.62
Output dim: 0, lower bound: -1.0491765, upper bound: 1.0486524
IS_A2_B1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 0, lower bound: -1.0487464, upper bound: 1.0507517
IS_A2_B1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 0, lower bound: -1.0487464, upper bound: 1.0507517
IS_A2_B1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 0, lower bound: -1.0483984, upper bound: 1.0508334
IS_A2_B1_B1_B1_A2_A2, status: Status.VERIFIED, split count: 6, time: 2.62
Output dim: 0, lower bound: -1.0433378, upper bound: 1.0457103
IS_A2_B1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 0, lower bound: -1.0484533, upper bound: 1.0507175
IS_A2_B1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 0, lower bound: -1.0484533, upper bound: 1.0507175
IS_A2_B1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 0, lower bound: -1.0484533, upper bound: 1.0507175
IS_A2_B1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 0, lower bound: -1.0484533, upper bound: 1.0507175
IS_A2_B1_B2_B1_A2_A1, status: Status.VERIFIED, split count: 6, time: 2.62
Output dim: 0, lower bound: -1.0478559, upper bound: 1.0494273
IS_A2_B1_B2_B1_A2_A2, status: Status.VERIFIED, split count: 6, time: 2.62
Output dim: 0, lower bound: -1.0431251, upper bound: 1.0444293
IS_A2_B1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.62
Output dim: 0, lower bound: -1.0481018, upper bound: 1.0497983
IS_A2_B1_B2_B2_A1_A2, status: Status.VERIFIED, split count: 6, time: 2.62
Output dim: 0, lower bound: -1.0481018, upper bound: 1.0493216
IS_A2_B1_B2_B2_A2_A1, status: Status.VERIFIED, split count: 6, time: 2.62
Output dim: 0, lower bound: -1.0481018, upper bound: 1.0493216
IS_A2_B1_B2_B2_A2_A2, status: Status.VERIFIED, split count: 6, time: 2.62
Output dim: 0, lower bound: -1.0481018, upper bound: 1.0493216
IS_A2_B2_B2_A2_B1_B1, status: Status.VERIFIED, split count: 6, time: 2.62
Output dim: 0, lower bound: -1.0425972, upper bound: 1.0428639
IS_A2_B2_B2_A2_B1_B2, status: Status.VERIFIED, split count: 6, time: 2.62
Output dim: 0, lower bound: -1.0257711, upper bound: 1.0272760

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1730340, 0.6613784, -0.1687949, 0.6562186, -0.8292526, 0.8301733
1: -0.3788329, 0.8342032, -0.3721583, 0.8298426, -1.2086755, 1.2063615
2: -0.3065768, 0.9387357, -0.2982762, 0.9297570, -1.2363338, 1.2370119
3: -0.7025846, 0.9433535, -0.6993915, 0.9345457, -1.6371303, 1.6427450
4: -0.5771155, 1.1376231, -0.5636945, 1.1223240, -1.6994395, 1.7013175

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513698, upper bound: 1.0499766
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4

Time for candidate selection: 5.01 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0495071, upper bound: 1.0490162
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0509838, upper bound: 1.0502371
time: 0.42 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1730340, 0.6613784, -0.2369063, 0.7675694, -0.9406034, 0.8982847
1: -0.3788329, 0.8342032, -0.4662368, 0.9817842, -1.3606172, 1.3004401
2: -0.3065768, 0.9387357, -0.3882421, 1.1031774, -1.4097543, 1.3269777
3: -0.7025846, 0.9433535, -0.8422803, 1.1139320, -1.8165166, 1.7856338
4: -0.5771155, 1.1376231, -0.7166221, 1.3037164, -1.8808320, 1.8542452

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513698, upper bound: 1.0499766
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 29
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 11

Time for candidate selection: 4.95 seconds

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0495071, upper bound: 1.0490162
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0509838, upper bound: 1.0502371
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.2542607, 0.8079799, -0.1591622, 0.6442558, -0.8985165, 0.9671421
1: -0.4890153, 1.0335444, -0.3579478, 0.8161792, -1.3051945, 1.3914922
2: -0.4136510, 1.1714401, -0.2839868, 0.9127556, -1.3264067, 1.4554269
3: -0.8771543, 1.1680876, -0.6818609, 0.9148057, -1.7919600, 1.8499484
4: -0.7573829, 1.3704841, -0.5420166, 1.0998335, -1.8572164, 1.9125007

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 4

Time for candidate selection: 4.56 seconds

### Candidate
type: A, layer: 3, pos: 31

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0488663, upper bound: 1.0472261
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0458304, upper bound: 1.0448165
time: 0.35 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.2542607, 0.8079799, -0.2369063, 0.7675694, -1.0218301, 1.0448864
1: -0.4890153, 1.0335444, -0.4662368, 0.9817842, -1.4707996, 1.4997813
2: -0.4136510, 1.1714401, -0.3882421, 1.1031774, -1.5168285, 1.5596823
3: -0.8771543, 1.1680876, -0.8422803, 1.1139320, -1.9910862, 2.0103679
4: -0.7573829, 1.3704841, -0.7166221, 1.3037164, -2.0610993, 2.0871062

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 29
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 11

Time for candidate selection: 4.60 seconds

### Candidate
type: A, layer: 3, pos: 31

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0462856, upper bound: 1.0459225
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0458304, upper bound: 1.0448165
time: 0.38 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1786847, 0.6695686, -0.1796174, 0.6729426, -0.8516273, 0.8491859
1: -0.3875105, 0.8442428, -0.3886608, 0.8546216, -1.2421322, 1.2329036
2: -0.3145856, 0.9516206, -0.3137678, 0.9577398, -1.2723253, 1.2653884
3: -0.7133602, 0.9566646, -0.7210382, 0.9618257, -1.6751859, 1.6777028
4: -0.5890352, 1.1542461, -0.5868647, 1.1561475, -1.7451826, 1.7411108

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0516123, upper bound: 1.0501336
time: 0.45 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0516123, upper bound: 1.0502515
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0516123, upper bound: 1.0502515
time: 0.44 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1786847, 0.6695686, -0.2148956, 0.7115479, -0.8902326, 0.8844643
1: -0.3875105, 0.8442428, -0.4401824, 0.8935100, -1.2810206, 1.2844253
2: -0.3145856, 0.9516206, -0.3639436, 1.0145798, -1.3291653, 1.3155642
3: -0.7133602, 0.9566646, -0.7847203, 1.0327761, -1.7461363, 1.7413849
4: -0.5890352, 1.1542461, -0.6645471, 1.2404692, -1.8295044, 1.8187933

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0516123, upper bound: 1.0503000
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0516123, upper bound: 1.0504071
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0516123, upper bound: 1.0504071
time: 0.46 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.2604521, 0.8180948, -0.1492833, 0.6228503, -0.8833025, 0.9673782
1: -0.4973333, 1.0453314, -0.3397722, 0.7941312, -1.2914646, 1.3851036
2: -0.4221133, 1.1859961, -0.2682760, 0.8903741, -1.3124874, 1.4542720
3: -0.8889325, 1.1832230, -0.6649222, 0.8833283, -1.7722607, 1.8481452
4: -0.7701943, 1.3884196, -0.5212010, 1.0665138, -1.8367081, 1.9096206

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499663, upper bound: 1.0499663
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499663, upper bound: 1.0499663
time: 0.40 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.2604521, 0.8180948, -0.1883637, 0.6663783, -0.9268304, 1.0064585
1: -0.4973333, 1.0453314, -0.3974727, 0.8419206, -1.3392539, 1.4428041
2: -0.4221133, 1.1859961, -0.3226977, 0.9568016, -1.3789148, 1.5086937
3: -0.8889325, 1.1832230, -0.7376361, 0.9610856, -1.8500180, 1.9208591
4: -0.7701943, 1.3884196, -0.6053913, 1.1609843, -1.9311786, 1.9938109

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499663, upper bound: 1.0501089
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499663, upper bound: 1.0501089
time: 0.42 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.2198625, 0.7168525, -0.2083114, 0.6935929, -0.9134554, 0.9251640
1: -0.4481273, 0.8946527, -0.4229872, 0.8801537, -1.3282809, 1.3176399
2: -0.3729504, 1.0201207, -0.3531615, 0.9850399, -1.3579904, 1.3732822
3: -0.7931444, 1.0421638, -0.7622833, 1.0002071, -1.7933514, 1.8044472
4: -0.6782902, 1.2525201, -0.6478740, 1.1869671, -1.8652574, 1.9003941

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_B1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513650, upper bound: 1.0506779
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513650, upper bound: 1.0506779
time: 0.41 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.2204756, 0.7195514, -0.1605757, 0.6465477, -0.8670233, 0.8801272
1: -0.4490335, 0.8986405, -0.3604908, 0.8188620, -1.2678955, 1.2591313
2: -0.3733449, 1.0248224, -0.2867638, 0.9154066, -1.2887515, 1.3115861
3: -0.7957247, 1.0452809, -0.6861663, 0.9179245, -1.7136492, 1.7314472
4: -0.6791479, 1.2565608, -0.5463678, 1.1035635, -1.7827115, 1.8029286

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_B1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0521540, upper bound: 1.0526937
time: 0.48 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0521540, upper bound: 1.0526937
time: 0.41 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.2058784, 0.7007015, -0.1796174, 0.6729426, -0.8788211, 0.8803189
1: -0.4285818, 0.8732541, -0.3886608, 0.8546216, -1.2832034, 1.2619150
2: -0.3534709, 0.9999138, -0.3137678, 0.9577398, -1.3112106, 1.3136816
3: -0.7660800, 1.0146086, -0.7210382, 0.9618257, -1.7279058, 1.7356468
4: -0.6508676, 1.2256274, -0.5868647, 1.1561475, -1.8070152, 1.8124921

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_A1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504068, upper bound: 1.0501585
time: 0.45 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501089, upper bound: 1.0501464
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501089, upper bound: 1.0501464
time: 0.45 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.2956260, 0.8480921, -0.1492833, 0.6228503, -0.9184762, 0.9973754
1: -0.5488452, 1.0708094, -0.3397722, 0.7941312, -1.3429763, 1.4105816
2: -0.4724841, 1.2222452, -0.2682760, 0.8903741, -1.3628582, 1.4905212
3: -0.9468486, 1.2486579, -0.6649222, 0.8833283, -1.8301768, 1.9135802
4: -0.8466773, 1.4592725, -0.5212010, 1.0665138, -1.9131911, 1.9804735

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501089, upper bound: 1.0501464
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501089, upper bound: 1.0501464
time: 0.44 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.2045429, 0.6960640, -0.2045429, 0.6960640, -0.9006069, 0.9006069
1: -0.4246404, 0.8731498, -0.4246404, 0.8731498, -1.2977902, 1.2977902
2: -0.3495150, 0.9898366, -0.3495150, 0.9898366, -1.3393517, 1.3393517
3: -0.7641593, 1.0075936, -0.7641593, 1.0075936, -1.7717528, 1.7717528
4: -0.6426289, 1.2076913, -0.6426289, 1.2076913, -1.8503202, 1.8503202

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0511125, upper bound: 1.0504385
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504744, upper bound: 1.0503219
time: 0.43 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2045429, 0.6960640, -0.2148956, 0.7115479, -0.9160908, 0.9109596
1: -0.4246404, 0.8731498, -0.4401824, 0.8935100, -1.3181505, 1.3133322
2: -0.3495150, 0.9898366, -0.3639436, 1.0145798, -1.3640947, 1.3537803
3: -0.7641593, 1.0075936, -0.7847203, 1.0327761, -1.7969353, 1.7923139
4: -0.6426289, 1.2076913, -0.6645471, 1.2404692, -1.8830981, 1.8722384

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0511125, upper bound: 1.0504983
time: 0.45 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504744, upper bound: 1.0503753
time: 0.52 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.2148956, 0.7115479, -0.2058784, 0.7007015, -0.9155971, 0.9174263
1: -0.4401824, 0.8935100, -0.4285818, 0.8732541, -1.3134365, 1.3220918
2: -0.3639436, 1.0145798, -0.3534709, 0.9999138, -1.3638574, 1.3680507
3: -0.7847203, 1.0327761, -0.7660800, 1.0146086, -1.7993289, 1.7988560
4: -0.6645471, 1.2404692, -0.6508676, 1.2256274, -1.8901746, 1.8913369

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504791, upper bound: 1.0508863
time: 0.50 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504791, upper bound: 1.0509103
time: 0.46 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.1895134, 0.6681194, -0.2956260, 0.8480921, -1.0376054, 0.9637454
1: -0.3992683, 0.8438303, -0.5488452, 1.0708094, -1.4700776, 1.3926754
2: -0.3244658, 0.9593039, -0.4724841, 1.2222452, -1.5467110, 1.4317880
3: -0.7399998, 0.9637969, -0.9468486, 1.2486579, -1.9886577, 1.9106455
4: -0.6080915, 1.1642671, -0.8466773, 1.4592725, -2.0673640, 2.0109444

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502161, upper bound: 1.0503624
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502161, upper bound: 1.0503624
time: 0.45 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.1730340, 0.6613784, -1.2387815, 2.5273116, -2.7003455, 1.9001598
1: -0.3788329, 0.8342032, -1.7145014, 2.7320752, -3.1109080, 2.5487046
2: -0.3065768, 0.9387357, -1.6877520, 3.0873444, -3.3939211, 2.6264877
3: -0.7025846, 0.9433535, -2.2010422, 3.5697715, -4.2723560, 3.1443958
4: -0.5771155, 1.1376231, -2.5692134, 3.6112943, -4.1884098, 3.7068365

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A1_B1_B1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507481, upper bound: 1.0473869
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 31

Time for candidate selection: 5.16 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A1_A1_B1_B1_A1

### Relational analysis result of IS_A1_B2_A1_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0487007, upper bound: 1.0472347
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_B1_A2

### Relational analysis result of IS_A1_B2_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0503731, upper bound: 1.0486617
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.1730340, 0.6613784, -1.2144685, 2.5487843, -2.7218180, 1.8758469
1: -0.3788329, 0.8342032, -1.6705327, 2.7996631, -3.1784959, 2.5047359
2: -0.3065768, 0.9387357, -1.6170993, 3.1658318, -3.4724085, 2.5558350
3: -0.7025846, 0.9433535, -2.2173247, 3.5713704, -4.2739549, 3.1606784
4: -0.5771155, 1.1376231, -2.4885974, 3.6392481, -4.2163639, 3.6262205

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A1_B1_B2_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507481, upper bound: 1.0473869
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: A, layer: 3, pos: 17
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 31

Time for candidate selection: 5.10 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A1_A1_B1_B2_A1

### Relational analysis result of IS_A1_B2_A1_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0487007, upper bound: 1.0472347
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_B2_A2

### Relational analysis result of IS_A1_B2_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0503731, upper bound: 1.0486617
time: 0.47 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.1534129, 0.6383394, -1.1764433, 2.4069278, -2.5603406, 1.8147827
1: -0.3508174, 0.8072805, -1.6424212, 2.6291804, -2.9799976, 2.4497018
2: -0.2782613, 0.9044772, -1.6098406, 2.9486027, -3.2268639, 2.5143180
3: -0.6693972, 0.9044048, -2.1219630, 3.4013817, -4.0707788, 3.0263679
4: -0.5349159, 1.0928698, -2.4546800, 3.4508541, -3.9857697, 3.5475497

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_A1_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0453164, upper bound: 1.0433378
time: 0.48 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_A1_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0453164, upper bound: 1.0433378
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.1905538, 0.6793638, -1.3571975, 2.6456168, -2.8361707, 2.0365615
1: -0.4057388, 0.8529248, -1.8233922, 2.8606629, -3.2664015, 2.6763170
2: -0.3302413, 0.9696531, -1.7856853, 3.2564571, -3.5866983, 2.7553384
3: -0.7378664, 0.9809170, -2.3469903, 3.7532561, -4.4911222, 3.3279073
4: -0.6153858, 1.1839089, -2.7108135, 3.8025465, -4.4179320, 3.8947225

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0497720, upper bound: 1.0485968
time: 0.44 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0497720, upper bound: 1.0486588
time: 0.39 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.47 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.47
Output dim: 0, lower bound: -1.0495071, upper bound: 1.0490162
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 0, lower bound: -1.0509838, upper bound: 1.0502371
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.47
Output dim: 0, lower bound: -1.0495071, upper bound: 1.0490162
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 0, lower bound: -1.0509838, upper bound: 1.0502371
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.47
Output dim: 0, lower bound: -1.0488663, upper bound: 1.0472261
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.47
Output dim: 0, lower bound: -1.0458304, upper bound: 1.0448165
IS_A1_B1_A1_B1_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 3.47
Output dim: 0, lower bound: -1.0462856, upper bound: 1.0459225
IS_A1_B1_A1_B1_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 3.47
Output dim: 0, lower bound: -1.0458304, upper bound: 1.0448165
IS_A1_B1_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 0, lower bound: -1.0516123, upper bound: 1.0502515
IS_A1_B1_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 0, lower bound: -1.0516123, upper bound: 1.0502515
IS_A1_B1_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 0, lower bound: -1.0516123, upper bound: 1.0504071
IS_A1_B1_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 0, lower bound: -1.0516123, upper bound: 1.0504071
IS_A1_B1_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 0, lower bound: -1.0499663, upper bound: 1.0499663
IS_A1_B1_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 0, lower bound: -1.0499663, upper bound: 1.0499663
IS_A1_B1_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 0, lower bound: -1.0499663, upper bound: 1.0501089
IS_A1_B1_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 0, lower bound: -1.0499663, upper bound: 1.0501089
IS_A1_B1_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 0, lower bound: -1.0513650, upper bound: 1.0506779
IS_A1_B1_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 0, lower bound: -1.0513650, upper bound: 1.0506779
IS_A1_B1_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 0, lower bound: -1.0521540, upper bound: 1.0526937
IS_A1_B1_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 0, lower bound: -1.0521540, upper bound: 1.0526937
IS_A1_B1_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 0, lower bound: -1.0501089, upper bound: 1.0501464
IS_A1_B1_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 0, lower bound: -1.0501089, upper bound: 1.0501464
IS_A1_B1_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 0, lower bound: -1.0501089, upper bound: 1.0501464
IS_A1_B1_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 0, lower bound: -1.0501089, upper bound: 1.0501464
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 0, lower bound: -1.0511125, upper bound: 1.0504385
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 0, lower bound: -1.0504744, upper bound: 1.0503219
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 0, lower bound: -1.0511125, upper bound: 1.0504983
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 0, lower bound: -1.0504744, upper bound: 1.0503753
IS_A1_B1_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 0, lower bound: -1.0504791, upper bound: 1.0508863
IS_A1_B1_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 0, lower bound: -1.0504791, upper bound: 1.0509103
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 0, lower bound: -1.0502161, upper bound: 1.0503624
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 0, lower bound: -1.0502161, upper bound: 1.0503624
IS_A1_B2_A1_A1_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.47
Output dim: 0, lower bound: -1.0487007, upper bound: 1.0472347
IS_A1_B2_A1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 0, lower bound: -1.0503731, upper bound: 1.0486617
IS_A1_B2_A1_A1_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.47
Output dim: 0, lower bound: -1.0487007, upper bound: 1.0472347
IS_A1_B2_A1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 0, lower bound: -1.0503731, upper bound: 1.0486617
IS_A1_B2_A1_A1_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.47
Output dim: 0, lower bound: -1.0453164, upper bound: 1.0433378
IS_A1_B2_A1_A1_B2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.47
Output dim: 0, lower bound: -1.0453164, upper bound: 1.0433378
IS_A1_B2_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 0, lower bound: -1.0497720, upper bound: 1.0485968
IS_A1_B2_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 0, lower bound: -1.0497720, upper bound: 1.0486588
IS_A2_B1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 0, lower bound: -1.0487464, upper bound: 1.0507517
IS_A2_B1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 0, lower bound: -1.0487464, upper bound: 1.0507517
IS_A2_B1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 0, lower bound: -1.0483984, upper bound: 1.0508334
IS_A2_B1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 0, lower bound: -1.0484533, upper bound: 1.0507175
IS_A2_B1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 0, lower bound: -1.0484533, upper bound: 1.0507175
IS_A2_B1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 0, lower bound: -1.0484533, upper bound: 1.0507175
IS_A2_B1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 0, lower bound: -1.0484533, upper bound: 1.0507175
IS_A2_B1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 0, lower bound: -1.0481018, upper bound: 1.0497983
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0250000, mid=0.0250000, abs_max=1.1883488893508911
rel_dist={0: [-1.0551075589159629, 1.0551075589159629]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 1150.98 seconds
