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
execution time: IAR + LP analysis = 1.60 + 1.07 = 2.67 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -1.0564887, upper bound: 1.0564887


# Binary Search by BASE starts (time budget: 1197.33 seconds, max iter: 100)

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
Binary search time: 47.45 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 1149.88 seconds

## Binary search (step 0) starts
Candidate diff: 0.1000000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0553496, upper bound: 1.0511832
time: 0.35 seconds

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

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
time: 0.34 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
time: 0.35 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -1.3271873, 2.7002649, -0.3035725, 0.8847764, -2.2119637, 3.0038373
1: -1.8266034, 2.9257355, -0.5660125, 1.0933844, -2.9199877, 3.4917479
2: -1.7866864, 3.3100519, -0.4826685, 1.2412479, -3.0279343, 3.7927201
3: -2.3538351, 3.8103127, -0.9617165, 1.2755736, -3.6294079, 4.7720289
4: -2.7129741, 3.8588223, -0.8354526, 1.4994075, -4.2123814, 4.6942744

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

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

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0549353, upper bound: 1.0510210
time: 0.36 seconds

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
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0549353, upper bound: 1.0510210
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535985, upper bound: 1.0510581
time: 0.37 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -1.3271873, 2.7002649, -0.2352601, 0.7538407, -2.0810280, 2.9355249
1: -1.8266034, 2.9257355, -0.4706453, 0.9488738, -2.7754772, 3.3963809
2: -1.7866864, 3.3100519, -0.3915833, 1.0723588, -2.8590453, 3.7016351
3: -2.3538351, 3.8103127, -0.8320177, 1.0878556, -3.4416907, 4.6423302
4: -2.7129741, 3.8588223, -0.7029035, 1.3031529, -4.0161266, 4.5617256

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0489732, upper bound: 1.0494879
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479269, upper bound: 1.0479269
time: 0.35 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -1.3271873, 2.7002649, -1.3271873, 2.7002649, -4.0274525, 4.0274525
1: -1.8266034, 2.9257355, -1.8266034, 2.9257355, -4.7523389, 4.7523384
2: -1.7866864, 3.3100519, -1.7866864, 3.3100519, -5.0967383, 5.0967379
3: -2.3538351, 3.8103127, -2.3538351, 3.8103127, -6.1641479, 6.1641479
4: -2.7129741, 3.8588223, -2.7129741, 3.8588223, -6.5717964, 6.5717964

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0489732, upper bound: 1.0494879
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479269, upper bound: 1.0479269
time: 0.38 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.34 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.34
Output dim: 0, lower bound: -1.0549353, upper bound: 1.0510210
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.34
Output dim: 0, lower bound: -1.0535985, upper bound: 1.0510581
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.34
Output dim: 0, lower bound: -1.0549353, upper bound: 1.0510210
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.34
Output dim: 0, lower bound: -1.0535985, upper bound: 1.0510581
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 2.34
Output dim: 0, lower bound: -1.0489732, upper bound: 1.0494879
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.34
Output dim: 0, lower bound: -1.0479269, upper bound: 1.0479269
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 2.34
Output dim: 0, lower bound: -1.0489732, upper bound: 1.0494879
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.34
Output dim: 0, lower bound: -1.0479269, upper bound: 1.0479269

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.2013956, 0.7023641, -0.2352601, 0.7538407, -0.9552363, 0.9376242
1: -0.4206367, 0.8831170, -0.4706453, 0.9488738, -1.3695104, 1.3537623
2: -0.3466128, 1.0003821, -0.3915833, 1.0723588, -1.4189715, 1.3919654
3: -0.7602279, 1.0080743, -0.8320177, 1.0878556, -1.8480835, 1.8400919
4: -0.6370696, 1.2147777, -0.7029035, 1.3031529, -1.9402225, 1.9176812

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539857, upper bound: 1.0539857
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539857, upper bound: 1.0539857
time: 0.36 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.2363607, 0.7416377, -0.2328624, 0.7488655, -0.9852261, 0.9745001
1: -0.4722191, 0.9222932, -0.4671082, 0.9423364, -1.4145554, 1.3894014
2: -0.3964722, 1.0577438, -0.3885702, 1.0657597, -1.4622319, 1.4463140
3: -0.8244337, 1.0810699, -0.8270060, 1.0809959, -1.9054296, 1.9080759
4: -0.7141775, 1.3009543, -0.6987019, 1.2962005, -2.0103781, 1.9996562

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539857, upper bound: 1.0540170
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539857, upper bound: 1.0540170
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.2013956, 0.7023641, -1.3271873, 2.7002649, -2.9016604, 2.0295515
1: -0.4206367, 0.8831170, -1.8266034, 2.9257355, -3.3463721, 2.7097204
2: -0.3466128, 1.0003821, -1.7866864, 3.3100519, -3.6566644, 2.7870684
3: -0.7602279, 1.0080743, -2.3538351, 3.8103127, -4.5705404, 3.3619094
4: -0.6370696, 1.2147777, -2.7129741, 3.8588223, -4.4958920, 3.9277518

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0549353, upper bound: 1.0507504
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0494336, upper bound: 1.0441275
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.2363607, 0.7416377, -1.3230309, 2.6886380, -2.9249988, 2.0646687
1: -0.4722191, 0.9222932, -1.8206375, 2.9125025, -3.3847218, 2.7429307
2: -0.3964722, 1.0577438, -1.7817578, 3.2965152, -3.6929874, 2.8395016
3: -0.8244337, 1.0810699, -2.3453972, 3.7946963, -4.6191301, 3.4264672
4: -0.7141775, 1.3009543, -2.7058206, 3.8440361, -4.5582137, 4.0067749

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513845, upper bound: 1.0496232
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513845, upper bound: 1.0510581
time: 0.41 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.83 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.83
Output dim: 0, lower bound: -1.0539857, upper bound: 1.0539857
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.83
Output dim: 0, lower bound: -1.0539857, upper bound: 1.0539857
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.83
Output dim: 0, lower bound: -1.0539857, upper bound: 1.0540170
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.83
Output dim: 0, lower bound: -1.0539857, upper bound: 1.0540170
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.83
Output dim: 0, lower bound: -1.0549353, upper bound: 1.0507504
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.83
Output dim: 0, lower bound: -1.0494336, upper bound: 1.0441275
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.83
Output dim: 0, lower bound: -1.0513845, upper bound: 1.0496232
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.83
Output dim: 0, lower bound: -1.0513845, upper bound: 1.0510581

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.2013956, 0.7023641, -0.2013956, 0.7023641, -0.9037597, 0.9037597
1: -0.4206367, 0.8831170, -0.4206367, 0.8831170, -1.3037536, 1.3037536
2: -0.3466128, 1.0003821, -0.3466128, 1.0003821, -1.3469949, 1.3469949
3: -0.7602279, 1.0080743, -0.7602279, 1.0080743, -1.7683022, 1.7683022
4: -0.6370696, 1.2147777, -0.6370696, 1.2147777, -1.8518473, 1.8518473

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0550041, upper bound: 1.0512581
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502805, upper bound: 1.0502805
time: 0.36 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.2013956, 0.7023641, -0.2363607, 0.7416377, -0.9430333, 0.9387248
1: -0.4206367, 0.8831170, -0.4722191, 0.9222932, -1.3429298, 1.3553361
2: -0.3466128, 1.0003821, -0.3964722, 1.0577438, -1.4043566, 1.3968543
3: -0.7602279, 1.0080743, -0.8244337, 1.0810699, -1.8412979, 1.8325080
4: -0.6370696, 1.2147777, -0.7141775, 1.3009543, -1.9380239, 1.9289553

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0550041, upper bound: 1.0518924
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502805, upper bound: 1.0508939
time: 0.41 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.2363607, 0.7416377, -0.2013956, 0.7023641, -0.9387248, 0.9430333
1: -0.4722191, 0.9222932, -0.4206367, 0.8831170, -1.3553361, 1.3429298
2: -0.3964722, 1.0577438, -0.3466128, 1.0003821, -1.3968543, 1.4043566
3: -0.8244337, 1.0810699, -0.7602279, 1.0080743, -1.8325080, 1.8412979
4: -0.7141775, 1.3009543, -0.6370696, 1.2147777, -1.9289553, 1.9380239

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
time: 0.37 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.2363607, 0.7416377, -0.2363607, 0.7416377, -0.9779984, 0.9779984
1: -0.4722191, 0.9222932, -0.4722191, 0.9222932, -1.3945123, 1.3945123
2: -0.3964722, 1.0577438, -0.3964722, 1.0577438, -1.4542160, 1.4542160
3: -0.8244337, 1.0810699, -0.8244337, 1.0810699, -1.9055036, 1.9055036
4: -0.7141775, 1.3009543, -0.7141775, 1.3009543, -2.0151320, 2.0151320

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1971632, 0.6977783, -1.2275646, 2.4774826, -2.6746454, 1.9253429
1: -0.4147021, 0.8778016, -1.7005773, 2.7084031, -3.1231050, 2.5783789
2: -0.3404469, 0.9934653, -1.6626792, 3.0481279, -3.3885748, 2.6561446
3: -0.7535037, 1.0000091, -2.1923127, 3.5047750, -4.2582788, 3.1923218
4: -0.6281362, 1.2057440, -2.5291798, 3.5632858, -4.1914220, 3.7349238

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0549353, upper bound: 1.0507291
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506076, upper bound: 1.0494973
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.2363032, 0.7415395, -1.2525434, 2.5990953, -2.8353977, 1.9940829
1: -0.4721391, 0.9221658, -1.7392304, 2.8298779, -3.3020170, 2.6613960
2: -0.3963988, 1.0575850, -1.6944914, 3.1925507, -3.5889492, 2.7520764
3: -0.8243142, 1.0809243, -2.2594891, 3.6576126, -4.4819260, 3.3404133
4: -0.7140632, 1.3007793, -2.5828991, 3.7129376, -4.4270005, 3.8836784

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513845, upper bound: 1.0496232
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513845, upper bound: 1.0496232
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.2363607, 0.7416377, -1.3095267, 2.6554937, -2.8918543, 2.0511644
1: -0.4722191, 0.9222932, -1.8045554, 2.8782458, -3.3504648, 2.7268486
2: -0.3964722, 1.0577438, -1.7664478, 3.2574177, -3.6538899, 2.8241916
3: -0.8244337, 1.0810699, -2.3240931, 3.7499304, -4.5743642, 3.4051630
4: -0.7141775, 1.3009543, -2.6825223, 3.8018088, -4.5159864, 3.9834766

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535985, upper bound: 1.0510581
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535985, upper bound: 1.0510581
time: 0.37 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.47 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 0, lower bound: -1.0550041, upper bound: 1.0512581
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 0, lower bound: -1.0502805, upper bound: 1.0502805
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 0, lower bound: -1.0550041, upper bound: 1.0518924
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 0, lower bound: -1.0502805, upper bound: 1.0508939
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 0, lower bound: -1.0549353, upper bound: 1.0507291
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 0, lower bound: -1.0506076, upper bound: 1.0494973
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 0, lower bound: -1.0513845, upper bound: 1.0496232
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 0, lower bound: -1.0513845, upper bound: 1.0496232
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 0, lower bound: -1.0535985, upper bound: 1.0510581
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.47
Output dim: 0, lower bound: -1.0535985, upper bound: 1.0510581

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1851257, 0.6783313, -0.2013956, 0.7023641, -0.8874898, 0.8797269
1: -0.3969364, 0.8526834, -0.4206367, 0.8831170, -1.2800534, 1.2733200
2: -0.3242711, 0.9644121, -0.3466128, 1.0003821, -1.3246531, 1.3110249
3: -0.7251614, 0.9704387, -0.7602279, 1.0080743, -1.7332357, 1.7306666
4: -0.6038694, 1.1718525, -0.6370696, 1.2147777, -1.8186471, 1.8089221

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502805, upper bound: 1.0502805
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502805, upper bound: 1.0502805
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.2674521, 0.8277271, -0.1939402, 0.6902832, -0.9577353, 1.0216672
1: -0.5075184, 1.0548140, -0.4086107, 0.8682333, -1.3757517, 1.4634247
2: -0.4327361, 1.2001319, -0.3356738, 0.9837908, -1.4165268, 1.5358057
3: -0.9017408, 1.1985904, -0.7464494, 0.9885674, -1.8903081, 1.9450397
4: -0.7864016, 1.4086894, -0.6209954, 1.1933510, -1.9797527, 2.0296848

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502805, upper bound: 1.0502805
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502805, upper bound: 1.0502805
time: 0.38 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1851257, 0.6783313, -0.2363607, 0.7416377, -0.9267634, 0.9146920
1: -0.3969364, 0.8526834, -0.4722191, 0.9222932, -1.3192296, 1.3249025
2: -0.3242711, 0.9644121, -0.3964722, 1.0577438, -1.3820149, 1.3608843
3: -0.7251614, 0.9704387, -0.8244337, 1.0810699, -1.8062314, 1.7948724
4: -0.6038694, 1.1718525, -0.7141775, 1.3009543, -1.9048238, 1.8860300

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502805, upper bound: 1.0502805
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502805, upper bound: 1.0502805
time: 0.41 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.2674521, 0.8277271, -0.2293040, 0.7306984, -0.9981505, 1.0570312
1: -0.5075184, 1.0548140, -0.4608361, 0.9096870, -1.4172055, 1.5156500
2: -0.4327361, 1.2001319, -0.3859614, 1.0431582, -1.4758942, 1.5860933
3: -0.9017408, 1.1985904, -0.8120781, 1.0629708, -1.9647115, 2.0106685
4: -0.7864016, 1.4086894, -0.6987422, 1.2806417, -2.0670433, 2.1074317

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502805, upper bound: 1.0502805
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508607, upper bound: 1.0508939
time: 0.42 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.2631326, 0.7698788, -0.2013307, 0.7022524, -0.9653850, 0.9712095
1: -0.5045100, 0.9689885, -0.4205441, 0.8829772, -1.3874872, 1.3895326
2: -0.4292829, 1.0956014, -0.3465242, 1.0002087, -1.4294916, 1.4421257
3: -0.8731403, 1.1296451, -0.7600935, 1.0078986, -1.8810389, 1.8897386
4: -0.7632787, 1.3396218, -0.6369393, 1.2145816, -1.9778603, 1.9765611

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0554253
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0554253
time: 0.36 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.2254990, 0.7289720, -0.2013956, 0.7023641, -0.9278631, 0.9303675
1: -0.4573010, 0.9075529, -0.4206367, 0.8831170, -1.3404180, 1.3281896
2: -0.3811812, 1.0387152, -0.3466128, 1.0003821, -1.3815633, 1.3853281
3: -0.8068940, 1.0590353, -0.7602279, 1.0080743, -1.8149683, 1.8192632
4: -0.6912529, 1.2759079, -0.6370696, 1.2147777, -1.9060307, 1.9129775

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0518924, upper bound: 1.0552379
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508939, upper bound: 1.0508607
time: 0.44 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.2631326, 0.7698788, -0.2363032, 0.7415395, -1.0046721, 1.0061820
1: -0.5045100, 0.9689885, -0.4721391, 0.9221658, -1.4266758, 1.4411275
2: -0.4292829, 1.0956014, -0.3963988, 1.0575850, -1.4868679, 1.4920002
3: -0.8731403, 1.1296451, -0.8243142, 1.0809243, -1.9540646, 1.9539593
4: -0.7632787, 1.3396218, -0.7140632, 1.3007793, -2.0640581, 2.0536849

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.2254990, 0.7289720, -0.2363607, 0.7416377, -0.9671367, 0.9653326
1: -0.4573010, 0.9075529, -0.4722191, 0.9222932, -1.3795942, 1.3797719
2: -0.3811812, 1.0387152, -0.3964722, 1.0577438, -1.4389250, 1.4351875
3: -0.8068940, 1.0590353, -0.8244337, 1.0810699, -1.8879640, 1.8834690
4: -0.6912529, 1.2759079, -0.7141775, 1.3009543, -1.9922073, 1.9900854

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1806452, 0.6735804, -1.2275646, 2.4774826, -2.6581278, 1.9011450
1: -0.3906853, 0.8471898, -1.7005773, 2.7084031, -3.0990884, 2.5477672
2: -0.3177781, 0.9573877, -1.6626792, 3.0481279, -3.3659060, 2.6200669
3: -0.7181361, 0.9620327, -2.1923127, 3.5047750, -4.2229114, 3.1543455
4: -0.5944355, 1.1625469, -2.5291798, 3.5632858, -4.1577215, 3.6917267

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0487347, upper bound: 1.0399864
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505847, upper bound: 1.0494973
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505847, upper bound: 1.0494973
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.2621900, 0.8206822, -1.2076378, 2.4348118, -2.6970012, 2.0283198
1: -0.5003384, 1.0463650, -1.6738672, 2.6622949, -3.1626334, 2.7202322
2: -0.4251498, 1.1893213, -1.6375039, 2.9983811, -3.4235303, 2.8268251
3: -0.8925821, 1.1872749, -2.1588161, 3.4454241, -4.3380060, 3.3460910
4: -0.7751930, 1.3956503, -2.4932671, 3.5066390, -4.2818317, 3.8889174

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505847, upper bound: 1.0494973
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505847, upper bound: 1.0494973
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.2631326, 0.7698788, -1.2525434, 2.5990953, -2.8622279, 2.0224223
1: -0.5045100, 0.9689885, -1.7392304, 2.8298779, -3.3343878, 2.7082188
2: -0.4292829, 1.0956014, -1.6944914, 3.1925507, -3.6218336, 2.7900929
3: -0.8731403, 1.1296451, -2.2594891, 3.6576126, -4.5307531, 3.3891342
4: -0.7632787, 1.3396218, -2.5828991, 3.7129376, -4.4762154, 3.9225209

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0495715
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0496232
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.2254990, 0.7289720, -1.2525434, 2.5990953, -2.8245943, 1.9815154
1: -0.4573010, 0.9075529, -1.7392304, 2.8298779, -3.2871790, 2.6467834
2: -0.3811812, 1.0387152, -1.6944914, 3.1925507, -3.5737319, 2.7332067
3: -0.8068940, 1.0590353, -2.2594891, 3.6576126, -4.4645052, 3.3185244
4: -0.6912529, 1.2759079, -2.5828991, 3.7129376, -4.4041905, 3.8588071

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0495715
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0496232
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.2631326, 0.7698788, -1.3095267, 2.6554937, -2.9186263, 2.0794055
1: -0.5045100, 0.9689885, -1.8045554, 2.8782458, -3.3827558, 2.7735438
2: -0.4292829, 1.0956014, -1.7664478, 3.2574177, -3.6867006, 2.8620491
3: -0.8731403, 1.1296451, -2.3240931, 3.7499304, -4.6230707, 3.4537382
4: -0.7632787, 1.3396218, -2.6825223, 3.8018088, -4.5650873, 4.0221443

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0500353
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0496232
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.2254990, 0.7289720, -1.3095267, 2.6554937, -2.8809927, 2.0384986
1: -0.4573010, 0.9075529, -1.8045554, 2.8782458, -3.3355470, 2.7121084
2: -0.3811812, 1.0387152, -1.7664478, 3.2574177, -3.6385989, 2.8051629
3: -0.8068940, 1.0590353, -2.3240931, 3.7499304, -4.5568242, 3.3831284
4: -0.6912529, 1.2759079, -2.6825223, 3.8018088, -4.4930620, 3.9584303

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0498831
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0498831
time: 0.39 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.34 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1.0502805, upper bound: 1.0502805
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1.0502805, upper bound: 1.0502805
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1.0502805, upper bound: 1.0502805
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1.0502805, upper bound: 1.0502805
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1.0502805, upper bound: 1.0502805
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1.0502805, upper bound: 1.0502805
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1.0502805, upper bound: 1.0502805
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1.0508607, upper bound: 1.0508939
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0554253
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0554253
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1.0518924, upper bound: 1.0552379
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1.0508939, upper bound: 1.0508607
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1.0505847, upper bound: 1.0494973
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1.0505847, upper bound: 1.0494973
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1.0505847, upper bound: 1.0494973
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1.0505847, upper bound: 1.0494973
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0495715
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0496232
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0495715
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0496232
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0500353
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0496232
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0498831
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.34
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0498831

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1851257, 0.6783313, -0.1851257, 0.6783313, -0.8634571, 0.8634571
1: -0.3969364, 0.8526834, -0.3969364, 0.8526834, -1.2496197, 1.2496197
2: -0.3242711, 0.9644121, -0.3242711, 0.9644121, -1.2886832, 1.2886832
3: -0.7251614, 0.9704387, -0.7251614, 0.9704387, -1.6956002, 1.6956002
4: -0.6038694, 1.1718525, -0.6038694, 1.1718525, -1.7757219, 1.7757219

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0537774, upper bound: 1.0511968
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0550041, upper bound: 1.0512581
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1851257, 0.6783313, -0.2674521, 0.8277271, -1.0128528, 0.9457834
1: -0.3969364, 0.8526834, -0.5075184, 1.0548140, -1.4517504, 1.3602018
2: -0.3242711, 0.9644121, -0.4327361, 1.2001319, -1.5244030, 1.3971481
3: -0.7251614, 0.9704387, -0.9017408, 1.1985904, -1.9237518, 1.8721795
4: -0.6038694, 1.1718525, -0.7864016, 1.4086894, -2.0125589, 1.9582541

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0537774, upper bound: 1.0511968
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0550041, upper bound: 1.0512581
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.2674521, 0.8277271, -0.1851257, 0.6783313, -0.9457834, 1.0128528
1: -0.5075184, 1.0548140, -0.3969364, 0.8526834, -1.3602018, 1.4517504
2: -0.4327361, 1.2001319, -0.3242711, 0.9644121, -1.3971481, 1.5244030
3: -0.9017408, 1.1985904, -0.7251614, 0.9704387, -1.8721795, 1.9237518
4: -0.7864016, 1.4086894, -0.6038694, 1.1718525, -1.9582541, 2.0125589

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501450, upper bound: 1.0502010
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501055, upper bound: 1.0501055
time: 0.38 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.2674521, 0.8277271, -0.2674521, 0.8277271, -1.0951792, 1.0951792
1: -0.5075184, 1.0548140, -0.5075184, 1.0548140, -1.5623324, 1.5623324
2: -0.4327361, 1.2001319, -0.4327361, 1.2001319, -1.6328681, 1.6328681
3: -0.9017408, 1.1985904, -0.9017408, 1.1985904, -2.1003313, 2.1003313
4: -0.7864016, 1.4086894, -0.7864016, 1.4086894, -2.1950910, 2.1950910

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501450, upper bound: 1.0502010
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501055, upper bound: 1.0501055
time: 0.42 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1851257, 0.6783313, -0.2122585, 0.7096910, -0.8948168, 0.8905898
1: -0.3969364, 0.8526834, -0.4381096, 0.8818882, -1.2788246, 1.2907929
2: -0.3242711, 0.9644121, -0.3631817, 1.0126708, -1.3369418, 1.3275938
3: -0.7251614, 0.9704387, -0.7779223, 1.0290604, -1.7542218, 1.7483611
4: -0.6038694, 1.1718525, -0.6656082, 1.2431200, -1.8469894, 1.8374606

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0551300, upper bound: 1.0518924
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0552379, upper bound: 1.0518924
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1851257, 0.6783313, -0.3022564, 0.8574660, -1.0425918, 0.9805877
1: -0.3969364, 0.8526834, -0.5588838, 1.0796481, -1.4765846, 1.4115672
2: -0.3242711, 0.9644121, -0.4826456, 1.2362378, -1.5605088, 1.4470577
3: -0.7251614, 0.9704387, -0.9591278, 1.2640674, -1.9892288, 1.9295666
4: -0.6038694, 1.1718525, -0.8620855, 1.4785453, -2.0824146, 2.0339379

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0551300, upper bound: 1.0518924
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0552379, upper bound: 1.0518924
time: 0.41 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.2674521, 0.8277271, -0.2122585, 0.7096910, -0.9771431, 1.0399857
1: -0.5075184, 1.0548140, -0.4381096, 0.8818882, -1.3894066, 1.4929236
2: -0.4327361, 1.2001319, -0.3631817, 1.0126708, -1.4454069, 1.5633136
3: -0.9017408, 1.1985904, -0.7779223, 1.0290604, -1.9308012, 1.9765127
4: -0.7864016, 1.4086894, -0.6656082, 1.2431200, -2.0295215, 2.0742974

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 6

Time for candidate selection: 2.40 seconds

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502084, upper bound: 1.0501708
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0434735, upper bound: 1.0478458
time: 0.43 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.2674521, 0.8277271, -0.3022564, 0.8574660, -1.1249182, 1.1299834
1: -0.5075184, 1.0548140, -0.5588838, 1.0796481, -1.5871665, 1.6136978
2: -0.4327361, 1.2001319, -0.4826456, 1.2362378, -1.6689739, 1.6827774
3: -0.9017408, 1.1985904, -0.9591278, 1.2640674, -2.1658082, 2.1577182
4: -0.7864016, 1.4086894, -0.8620855, 1.4785453, -2.2649469, 2.2707748

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
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
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 6

Time for candidate selection: 2.42 seconds

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502084, upper bound: 1.0501708
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0434735, upper bound: 1.0478612
time: 0.44 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.2631326, 0.7698788, -0.2350464, 0.7415904, -1.0047231, 1.0049253
1: -0.5045100, 0.9689885, -0.4636297, 0.9372106, -1.4417207, 1.4326181
2: -0.4292829, 1.0956014, -0.3912313, 1.0568323, -1.4861152, 1.4868327
3: -0.8731403, 1.1296451, -0.8212245, 1.0728223, -1.9459627, 1.9508696
4: -0.7632787, 1.3396218, -0.7068326, 1.2774920, -2.0407708, 2.0464544

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0537152, upper bound: 1.0554253
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0528525, upper bound: 1.0547487
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 4

Time for candidate selection: 3.28 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0550631
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0551583
time: 0.40 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.2631326, 0.7698788, -0.1909873, 0.6899648, -0.9530973, 0.9608661
1: -0.5045100, 0.9689885, -0.4062611, 0.8681383, -1.3726482, 1.3752496
2: -0.4292829, 1.0956014, -0.3319388, 0.9815091, -1.4107921, 1.4275403
3: -0.8731403, 1.1296451, -0.7429183, 0.9865521, -1.8596925, 1.8725634
4: -0.7632787, 1.3396218, -0.6151925, 1.1911318, -1.9544106, 1.9548143

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0537152, upper bound: 1.0554253
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0528525, upper bound: 1.0547487
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 4

Time for candidate selection: 3.52 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0550631
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0551583
time: 0.37 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.2254990, 0.7289720, -0.1851257, 0.6783313, -0.9038303, 0.9140977
1: -0.4573010, 0.9075529, -0.3969364, 0.8526834, -1.3099844, 1.3044894
2: -0.3811812, 1.0387152, -0.3242711, 0.9644121, -1.3455933, 1.3629863
3: -0.8068940, 1.0590353, -0.7251614, 0.9704387, -1.7773328, 1.7841967
4: -0.6912529, 1.2759079, -0.6038694, 1.1718525, -1.8631054, 1.8797773

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508939, upper bound: 1.0508607
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508939, upper bound: 1.0508607
time: 0.48 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.2180328, 0.7178377, -0.2674521, 0.8277271, -1.0457599, 0.9852898
1: -0.4454186, 0.8947487, -0.5075184, 1.0548140, -1.5002326, 1.4022671
2: -0.3700606, 1.0241183, -0.4327361, 1.2001319, -1.5701925, 1.4568543
3: -0.7941274, 1.0404365, -0.9017408, 1.1985904, -1.9927177, 1.9421773
4: -0.6749226, 1.2551465, -0.7864016, 1.4086894, -2.0836120, 2.0415483

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508939, upper bound: 1.0508607
time: 0.45 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508939, upper bound: 1.0508607
time: 0.41 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.2631326, 0.7698788, -0.2631326, 0.7698788, -1.0330114, 1.0330114
1: -0.5045100, 0.9689885, -0.5045100, 0.9689885, -1.4734986, 1.4734986
2: -0.4292829, 1.0956014, -0.4292829, 1.0956014, -1.5248843, 1.5248843
3: -0.8731403, 1.1296451, -0.8731403, 1.1296451, -2.0027854, 2.0027854
4: -0.7632787, 1.3396218, -0.7632787, 1.3396218, -2.1029005, 2.1029005

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 4

Time for candidate selection: 2.55 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0536725
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0537102
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2631326, 0.7698788, -0.2254990, 0.7289720, -0.9921045, 0.9953778
1: -0.5045100, 0.9689885, -0.4573010, 0.9075529, -1.4120629, 1.4262896
2: -0.4292829, 1.0956014, -0.3811812, 1.0387152, -1.4679981, 1.4767827
3: -0.8731403, 1.1296451, -0.8068940, 1.0590353, -1.9321756, 1.9365392
4: -0.7632787, 1.3396218, -0.6912529, 1.2759079, -2.0391865, 2.0308747

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 4

Time for candidate selection: 2.53 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0536725
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0537102
time: 0.35 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.2254990, 0.7289720, -0.2631326, 0.7698788, -0.9953778, 0.9921045
1: -0.4573010, 0.9075529, -0.5045100, 0.9689885, -1.4262896, 1.4120629
2: -0.3811812, 1.0387152, -0.4292829, 1.0956014, -1.4767827, 1.4679981
3: -0.8068940, 1.0590353, -0.8731403, 1.1296451, -1.9365392, 1.9321756
4: -0.6912529, 1.2759079, -0.7632787, 1.3396218, -2.0308747, 2.0391865

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 4

Time for candidate selection: 2.55 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0536198
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0537102
time: 0.41 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.2254990, 0.7289720, -0.2254990, 0.7289720, -0.9544709, 0.9544709
1: -0.4573010, 0.9075529, -0.4573010, 0.9075529, -1.3648539, 1.3648539
2: -0.3811812, 1.0387152, -0.3811812, 1.0387152, -1.4198965, 1.4198965
3: -0.8068940, 1.0590353, -0.8068940, 1.0590353, -1.8659294, 1.8659294
4: -0.6912529, 1.2759079, -0.6912529, 1.2759079, -1.9671608, 1.9671608

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 4

Time for candidate selection: 2.70 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0536198
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0537102
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1806452, 0.6735804, -1.2136881, 2.4463463, -2.6269915, 1.8872685
1: -0.3906853, 0.8471898, -1.6815057, 2.6714954, -3.0621808, 2.5286956
2: -0.3177781, 0.9573877, -1.6466799, 3.0085113, -3.3262894, 2.6040676
3: -0.7181361, 0.9620327, -2.1654463, 3.4600346, -4.1781707, 3.1274791
4: -0.5944355, 1.1625469, -2.5056057, 3.5195577, -4.1139932, 3.6681526

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0549351, upper bound: 1.0507291
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0549350, upper bound: 1.0507214
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1806452, 0.6735804, -1.1447992, 2.3595920, -2.5402372, 1.8183796
1: -0.3906853, 0.8471898, -1.5842650, 2.6178081, -3.0084934, 2.4314549
2: -0.3177781, 0.9573877, -1.5295095, 2.9590945, -3.2768726, 2.4868972
3: -0.7181361, 0.9620327, -2.1012015, 3.3203290, -4.0384645, 3.0632343
4: -0.5944355, 1.1625469, -2.3559012, 3.4138608, -4.0082960, 3.5184481

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0549351, upper bound: 1.0507291
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0549350, upper bound: 1.0507214
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.2621900, 0.8206822, -1.2136881, 2.4463463, -2.7085361, 2.0343704
1: -0.5003384, 1.0463650, -1.6815057, 2.6714954, -3.1718340, 2.7278707
2: -0.4251498, 1.1893213, -1.6466799, 3.0085113, -3.4336610, 2.8360012
3: -0.8925821, 1.1872749, -2.1654463, 3.4600346, -4.3526168, 3.3527212
4: -0.7751930, 1.3956503, -2.5056057, 3.5195577, -4.2947507, 3.9012561

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505847, upper bound: 1.0494973
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505847, upper bound: 1.0494973
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.2621900, 0.8206822, -1.1447992, 2.3595920, -2.6217816, 1.9654814
1: -0.5003384, 1.0463650, -1.5842650, 2.6178081, -3.1181464, 2.6306300
2: -0.4251498, 1.1893213, -1.5295095, 2.9590945, -3.3842444, 2.7188308
3: -0.8925821, 1.1872749, -2.1012015, 3.3203290, -4.2129111, 3.2884765
4: -0.7751930, 1.3956503, -2.3559012, 3.4138608, -4.1890535, 3.7515516

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505847, upper bound: 1.0494973
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505847, upper bound: 1.0494973
time: 0.45 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.2631326, 0.7698788, -1.2317941, 2.5485647, -2.8116972, 2.0016730
1: -0.5045100, 0.9689885, -1.7097392, 2.7696524, -3.2741623, 2.6787276
2: -0.4292829, 1.0956014, -1.6688566, 3.1283512, -3.5576341, 2.7644582
3: -0.8731403, 1.1296451, -2.2169631, 3.5870750, -4.4602156, 3.3466082
4: -0.7632787, 1.3396218, -2.5448000, 3.6422343, -4.4055128, 3.8844218

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 4

Time for candidate selection: 2.43 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0491910
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0492286
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.2631326, 0.7698788, -1.3011413, 2.5799091, -2.8430414, 2.0710201
1: -0.5045100, 0.9689885, -1.7600503, 2.8037138, -3.3082237, 2.7290387
2: -0.4292829, 1.0956014, -1.7155523, 3.1846526, -3.6139355, 2.8111539
3: -0.8731403, 1.1296451, -2.2817159, 3.6491742, -4.5223141, 3.4113610
4: -0.7632787, 1.3396218, -2.6121106, 3.7078054, -4.4710827, 3.9517324

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 4

Time for candidate selection: 2.40 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0492215
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0492755
time: 0.34 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.2254990, 0.7289720, -1.2317941, 2.5485647, -2.7740636, 1.9607661
1: -0.4573010, 0.9075529, -1.7097392, 2.7696524, -3.2269535, 2.6172922
2: -0.3811812, 1.0387152, -1.6688566, 3.1283512, -3.5095320, 2.7075720
3: -0.8068940, 1.0590353, -2.2169631, 3.5870750, -4.3939691, 3.2759984
4: -0.6912529, 1.2759079, -2.5448000, 3.6422343, -4.3334870, 3.8207078

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 4

Time for candidate selection: 2.48 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0491383
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0492286
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.2254990, 0.7289720, -1.3011413, 2.5799091, -2.8054080, 2.0301132
1: -0.4573010, 0.9075529, -1.7600503, 2.8037138, -3.2610145, 2.6676033
2: -0.3811812, 1.0387152, -1.7155523, 3.1846526, -3.5658336, 2.7542677
3: -0.8068940, 1.0590353, -2.2817159, 3.6491742, -4.4560671, 3.3407512
4: -0.6912529, 1.2759079, -2.6121106, 3.7078054, -4.3990583, 3.8880186

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 4

Time for candidate selection: 2.45 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0491851
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0492755
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.2631326, 0.7698788, -1.2890360, 2.6080267, -2.8711593, 2.0589149
1: -0.5045100, 0.9689885, -1.7757368, 2.8224382, -3.3269482, 2.7447252
2: -0.4292829, 1.0956014, -1.7412896, 3.1970921, -3.6263750, 2.8368912
3: -0.8731403, 1.1296451, -2.2828507, 3.6829684, -4.5561085, 3.4124959
4: -0.7632787, 1.3396218, -2.6450953, 3.7347975, -4.4980764, 3.9847171

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0516793, upper bound: 1.0491306
time: 0.45 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0400711, upper bound: 1.0397066
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512876, upper bound: 1.0492136
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 4

Time for candidate selection: 3.85 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0496908
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0497285
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2631326, 0.7698788, -1.3504868, 2.6251640, -2.8882966, 2.1203656
1: -0.5045100, 0.9689885, -1.8179705, 2.8394029, -3.3439131, 2.7869589
2: -0.4292829, 1.0956014, -1.7806034, 3.2355962, -3.6648791, 2.8762050
3: -0.8731403, 1.1296451, -2.3383503, 3.7266910, -4.5998311, 3.4679954
4: -0.7632787, 1.3396218, -2.7018437, 3.7814958, -4.5447741, 4.0414658

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0516793, upper bound: 1.0500775
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0400711, upper bound: 1.0397066
time: 0.53 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512876, upper bound: 1.0493899
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 4

Time for candidate selection: 3.92 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0499615
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0499680
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.2254990, 0.7289720, -1.2890360, 2.6080267, -2.8335257, 2.0180080
1: -0.4573010, 0.9075529, -1.7757368, 2.8224382, -3.2797394, 2.6832898
2: -0.3811812, 1.0387152, -1.7412896, 3.1970921, -3.5782728, 2.7800050
3: -0.8068940, 1.0590353, -2.2828507, 3.6829684, -4.4898624, 3.3418860
4: -0.6912529, 1.2759079, -2.6450953, 3.7347975, -4.4260502, 3.9210033

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469137, upper bound: 1.0480211
time: 0.44 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0486550, upper bound: 1.0485712
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.2254990, 0.7289720, -1.3504868, 2.6251640, -2.8506629, 2.0794587
1: -0.4573010, 0.9075529, -1.8179705, 2.8394029, -3.2967038, 2.7255235
2: -0.3811812, 1.0387152, -1.7806034, 3.2355962, -3.6167774, 2.8193188
3: -0.8068940, 1.0590353, -2.3383503, 3.7266910, -4.5335836, 3.3973856
4: -0.6912529, 1.2759079, -2.7018437, 3.7814958, -4.4727488, 3.9777517

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0469137, upper bound: 1.0506330
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0486551, upper bound: 1.0506391
time: 0.43 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 2.63 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0537774, upper bound: 1.0511968
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0550041, upper bound: 1.0512581
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0537774, upper bound: 1.0511968
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0550041, upper bound: 1.0512581
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0501450, upper bound: 1.0502010
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0501055, upper bound: 1.0501055
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0501450, upper bound: 1.0502010
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0501055, upper bound: 1.0501055
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0551300, upper bound: 1.0518924
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0552379, upper bound: 1.0518924
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0551300, upper bound: 1.0518924
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0552379, upper bound: 1.0518924
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0502084, upper bound: 1.0501708
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0434735, upper bound: 1.0478458
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0502084, upper bound: 1.0501708
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0434735, upper bound: 1.0478612
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0550631
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0551583
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0550631
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0551583
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0508939, upper bound: 1.0508607
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0508939, upper bound: 1.0508607
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0508939, upper bound: 1.0508607
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0508939, upper bound: 1.0508607
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0536725
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0537102
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0536725
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0537102
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0536198
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0537102
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0536198
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0537102
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0549351, upper bound: 1.0507291
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0549350, upper bound: 1.0507214
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0549351, upper bound: 1.0507291
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0549350, upper bound: 1.0507214
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0505847, upper bound: 1.0494973
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0505847, upper bound: 1.0494973
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0505847, upper bound: 1.0494973
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0505847, upper bound: 1.0494973
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0491910
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0492286
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0492215
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0492755
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0491383
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0492286
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0491851
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0492755
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0496908
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0497285
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0499615
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0499680
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0469137, upper bound: 1.0480211
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0486550, upper bound: 1.0485712
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0469137, upper bound: 1.0506330
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.63
Output dim: 0, lower bound: -1.0486551, upper bound: 1.0506391

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.2096419, 0.7101620, -0.1850603, 0.6782190, -0.8878608, 0.8952224
1: -0.4267260, 0.8983275, -0.3968441, 0.8525414, -1.2792674, 1.2951716
2: -0.3558276, 1.0129061, -0.3241817, 0.9642387, -1.3200662, 1.3370878
3: -0.7740620, 1.0201761, -0.7250257, 0.9702651, -1.7443271, 1.7452017
4: -0.6563050, 1.2186592, -0.6037393, 1.1716568, -1.8279618, 1.8223984

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0555443, upper bound: 1.0555443
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0555443, upper bound: 1.0555443
time: 0.42 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1746585, 0.6661327, -0.1851257, 0.6783313, -0.8529898, 0.8512585
1: -0.3825408, 0.8379728, -0.3969364, 0.8526834, -1.2352242, 1.2349092
2: -0.3095305, 0.9459749, -0.3242711, 0.9644121, -1.2739426, 1.2702460
3: -0.7079530, 0.9492630, -0.7251614, 0.9704387, -1.6783917, 1.6744244
4: -0.5819045, 1.1485283, -0.6038694, 1.1718525, -1.7537570, 1.7523978

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0555443, upper bound: 1.0555617
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0555443, upper bound: 1.0555617
time: 0.42 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.2096419, 0.7101620, -0.2673977, 0.8276447, -1.0372865, 0.9775597
1: -0.4267260, 0.8983275, -0.5074411, 1.0547137, -1.4814397, 1.4057686
2: -0.3558276, 1.0129061, -0.4326628, 1.2000061, -1.5558337, 1.4455689
3: -0.7740620, 1.0201761, -0.9016315, 1.1984571, -1.9725192, 1.9218075
4: -0.6563050, 1.2186592, -0.7862870, 1.4085369, -2.0648417, 2.0049462

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0537774, upper bound: 1.0509606
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0533379, upper bound: 1.0508971
time: 0.41 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1746585, 0.6661327, -0.2674521, 0.8277271, -1.0023856, 0.9335849
1: -0.3825408, 0.8379728, -0.5075184, 1.0548140, -1.4373548, 1.3454912
2: -0.3095305, 0.9459749, -0.4327361, 1.2001319, -1.5096624, 1.3787110
3: -0.7079530, 0.9492630, -0.9017408, 1.1985904, -1.9065434, 1.8510039
4: -0.5819045, 1.1485283, -0.7864016, 1.4086894, -1.9905939, 1.9349300

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0550041, upper bound: 1.0510182
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0548845, upper bound: 1.0509748
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1829112, 0.7200209, -0.1806452, 0.6735804, -0.8564916, 0.9006661
1: -0.3864989, 0.9190384, -0.3906853, 0.8471898, -1.2336888, 1.3097237
2: -0.3115277, 1.0421524, -0.3177781, 0.9573877, -1.2689154, 1.3599305
3: -0.7443157, 1.0138530, -0.7181361, 0.9620327, -1.7063484, 1.7319891
4: -0.6034802, 1.2099588, -0.5944355, 1.1625469, -1.7660271, 1.8043942

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0509606, upper bound: 1.0537774
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0510182, upper bound: 1.0550041
time: 0.43 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.2494289, 0.7944719, -0.1851257, 0.6783313, -0.9277602, 0.9795976
1: -0.4813044, 1.0119214, -0.3969364, 0.8526834, -1.3339877, 1.4088578
2: -0.4077803, 1.1496273, -0.3242711, 0.9644121, -1.3721924, 1.4738984
3: -0.8642187, 1.1482675, -0.7251614, 0.9704387, -1.8346574, 1.8734289
4: -0.7469358, 1.3539220, -0.6038694, 1.1718525, -1.9187883, 1.9577914

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508971, upper bound: 1.0533379
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0509748, upper bound: 1.0548845
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1829112, 0.7200209, -0.2621900, 0.8206822, -1.0035934, 0.9822109
1: -0.3864989, 0.9190384, -0.5003384, 1.0463650, -1.4328640, 1.4193769
2: -0.3115277, 1.0421524, -0.4251498, 1.1893213, -1.5008490, 1.4673022
3: -0.7443157, 1.0138530, -0.8925821, 1.1872749, -1.9315907, 1.9064350
4: -0.6034802, 1.2099588, -0.7751930, 1.3956503, -1.9991305, 1.9851518

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501055, upper bound: 1.0501055
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501055, upper bound: 1.0501055
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.2494289, 0.7944719, -0.2674521, 0.8277271, -1.0771561, 1.0619240
1: -0.4813044, 1.0119214, -0.5075184, 1.0548140, -1.5361184, 1.5194398
2: -0.4077803, 1.1496273, -0.4327361, 1.2001319, -1.6079122, 1.5823634
3: -0.8642187, 1.1482675, -0.9017408, 1.1985904, -2.0628090, 2.0500083
4: -0.7469358, 1.3539220, -0.7864016, 1.4086894, -2.1556253, 2.1403236

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501055, upper bound: 1.0501055
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501055, upper bound: 1.0501055
time: 0.45 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.2096419, 0.7101620, -0.2122009, 0.7095919, -0.9192337, 0.9223629
1: -0.4267260, 0.8983275, -0.4380294, 0.8817595, -1.3084855, 1.3363569
2: -0.3558276, 1.0129061, -0.3631084, 1.0125130, -1.3683406, 1.3760145
3: -0.7740620, 1.0201761, -0.7778029, 1.0289152, -1.8029771, 1.7979790
4: -0.6563050, 1.2186592, -0.6654948, 1.2429459, -1.8992509, 1.8841540

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0554253, upper bound: 1.0539330
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0554253, upper bound: 1.0539330
time: 0.38 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1746585, 0.6661327, -0.2122585, 0.7096910, -0.8843495, 0.8783913
1: -0.3825408, 0.8379728, -0.4381096, 0.8818882, -1.2644290, 1.2760824
2: -0.3095305, 0.9459749, -0.3631817, 1.0126708, -1.3222013, 1.3091567
3: -0.7079530, 0.9492630, -0.7779223, 1.0290604, -1.7370133, 1.7271854
4: -0.5819045, 1.1485283, -0.6656082, 1.2431200, -1.8250245, 1.8141365

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0554253, upper bound: 1.0539330
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0554253, upper bound: 1.0539330
time: 0.37 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.2096419, 0.7101620, -0.3021868, 0.8573557, -1.0669975, 1.0123489
1: -0.4267260, 0.8983275, -0.5587928, 1.0795094, -1.5062354, 1.4571202
2: -0.3558276, 1.0129061, -0.4825497, 1.2360641, -1.5918916, 1.4954557
3: -0.7740620, 1.0201761, -0.9589926, 1.2638978, -2.0379598, 1.9791687
4: -0.6563050, 1.2186592, -0.8619374, 1.4783511, -2.1346560, 2.0805964

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0551300, upper bound: 1.0516505
time: 0.45 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0545562, upper bound: 1.0511174
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0544140, upper bound: 1.0508131
time: 0.44 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1746585, 0.6661327, -0.3022564, 0.8574660, -1.0321245, 0.9683892
1: -0.3825408, 0.8379728, -0.5588838, 1.0796481, -1.4621890, 1.3968565
2: -0.3095305, 0.9459749, -0.4826456, 1.2362378, -1.5457683, 1.4286206
3: -0.7079530, 0.9492630, -0.9591278, 1.2640674, -1.9720204, 1.9083909
4: -0.5819045, 1.1485283, -0.8620855, 1.4785453, -2.0604498, 2.0106139

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0552379, upper bound: 1.0516505
time: 0.46 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0549649, upper bound: 1.0511174
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0550041, upper bound: 1.0508972
time: 0.42 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.2343636, 0.7751515, -0.2122585, 0.7096910, -0.9440545, 0.9874101
1: -0.4597959, 0.9859009, -0.4381096, 0.8818882, -1.3416841, 1.4240105
2: -0.3856044, 1.1241981, -0.3631817, 1.0126708, -1.3982751, 1.4873798
3: -0.8403842, 1.1136440, -0.7779223, 1.0290604, -1.8694446, 1.8915663
4: -0.7135836, 1.3228402, -0.6656082, 1.2431200, -1.9567037, 1.9884484

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499481, upper bound: 1.0504579
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502251, upper bound: 1.0509033
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.2343636, 0.7751515, -0.3022564, 0.8574660, -1.0918295, 1.0774078
1: -0.4597959, 0.9859009, -0.5588838, 1.0796481, -1.5394440, 1.5447848
2: -0.3856044, 1.1241981, -0.4826456, 1.2362378, -1.6218421, 1.6068437
3: -0.8403842, 1.1136440, -0.9591278, 1.2640674, -2.1044517, 2.0727718
4: -0.7135836, 1.3228402, -0.8620855, 1.4785453, -2.1921289, 2.1849256

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500625, upper bound: 1.0500336
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499013, upper bound: 1.0497225
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0494542, upper bound: 1.0483698
time: 0.48 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1981437, 0.6735660, -0.2350464, 0.7415904, -0.9397342, 0.9086125
1: -0.4075772, 0.8701689, -0.4636297, 0.9372106, -1.3447878, 1.3337986
2: -0.3290880, 0.9651487, -0.3912313, 1.0568323, -1.3859203, 1.3563800
3: -0.7565911, 0.9793513, -0.8212245, 1.0728223, -1.8294134, 1.8005757
4: -0.6072877, 1.1647243, -0.7068326, 1.2774920, -1.8847797, 1.8715570

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 4

Time for candidate selection: 2.66 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0550223
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0550282
time: 0.38 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.2620909, 0.7696464, -0.2350464, 0.7415904, -1.0036813, 1.0046928
1: -0.5045343, 0.9919055, -0.4636297, 0.9372106, -1.4417449, 1.4555352
2: -0.4200620, 1.0960678, -0.3912313, 1.0568323, -1.4768944, 1.4872991
3: -0.8876588, 1.1395187, -0.8212245, 1.0728223, -1.9604812, 1.9607432
4: -0.7470124, 1.3261604, -0.7068326, 1.2774920, -2.0245044, 2.0329931

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 4

Time for candidate selection: 2.65 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0551914
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0551945
time: 0.43 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1981437, 0.6735660, -0.1909873, 0.6899648, -0.8881085, 0.8645533
1: -0.4075772, 0.8701689, -0.4062611, 0.8681383, -1.2757154, 1.2764300
2: -0.3290880, 0.9651487, -0.3319388, 0.9815091, -1.3105972, 1.2970874
3: -0.7565911, 0.9793513, -0.7429183, 0.9865521, -1.7431432, 1.7222695
4: -0.6072877, 1.1647243, -0.6151925, 1.1911318, -1.7984195, 1.7799169

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 4

Time for candidate selection: 2.65 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0534783, upper bound: 1.0549755
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0534783, upper bound: 1.0550006
time: 0.36 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.2620909, 0.7696464, -0.1909873, 0.6899648, -0.9520556, 0.9606337
1: -0.5045343, 0.9919055, -0.4062611, 0.8681383, -1.3726726, 1.3981667
2: -0.4200620, 1.0960678, -0.3319388, 0.9815091, -1.4015712, 1.4280066
3: -0.8876588, 1.1395187, -0.7429183, 0.9865521, -1.8742108, 1.8824370
4: -0.7470124, 1.3261604, -0.6151925, 1.1911318, -1.9381442, 1.9413530

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 4

Time for candidate selection: 2.58 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0534783, upper bound: 1.0551410
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0534783, upper bound: 1.0551583
time: 0.42 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.2013567, 0.6970940, -0.1851257, 0.6783313, -0.8796880, 0.8822198
1: -0.4231896, 0.8673437, -0.3969364, 0.8526834, -1.2758729, 1.2642801
2: -0.3478338, 0.9940906, -0.3242711, 0.9644121, -1.3122458, 1.3183616
3: -0.7603652, 1.0071222, -0.7251614, 0.9704387, -1.7308040, 1.7322836
4: -0.6426137, 1.2188928, -0.6038694, 1.1718525, -1.8144662, 1.8227623

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0518924, upper bound: 1.0551300
time: 0.48 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0518924, upper bound: 1.0552379
time: 0.42 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.2909008, 0.8438774, -0.1851257, 0.6783313, -0.9692321, 1.0290031
1: -0.5446191, 1.0634959, -0.3969364, 0.8526834, -1.3973025, 1.4604323
2: -0.4665691, 1.2150321, -0.3242711, 0.9644121, -1.4309812, 1.5393032
3: -0.9410769, 1.2407138, -0.7251614, 0.9704387, -1.9115157, 1.9658753
4: -0.8382176, 1.4520541, -0.6038694, 1.1718525, -2.0100701, 2.0559235

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0518924, upper bound: 1.0551300
time: 0.51 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0518924, upper bound: 1.0552379
time: 0.43 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.2013567, 0.6970940, -0.2674521, 0.8277271, -1.0290837, 0.9645461
1: -0.4231896, 0.8673437, -0.5075184, 1.0548140, -1.4780036, 1.3748621
2: -0.3478338, 0.9940906, -0.4327361, 1.2001319, -1.5479656, 1.4268267
3: -0.7603652, 1.0071222, -0.9017408, 1.1985904, -1.9589556, 1.9088629
4: -0.6426137, 1.2188928, -0.7864016, 1.4086894, -2.0513031, 2.0052943

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 6

Time for candidate selection: 2.59 seconds

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501708, upper bound: 1.0502084
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474348, upper bound: 1.0469382
time: 0.42 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.2909008, 0.8438774, -0.2674521, 0.8277271, -1.1186279, 1.1113296
1: -0.5446191, 1.0634959, -0.5075184, 1.0548140, -1.5994332, 1.5710143
2: -0.4665691, 1.2150321, -0.4327361, 1.2001319, -1.6667011, 1.6477683
3: -0.9410769, 1.2407138, -0.9017408, 1.1985904, -2.1396673, 2.1424546
4: -0.8382176, 1.4520541, -0.7864016, 1.4086894, -2.2469070, 2.2384558

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 6

Time for candidate selection: 2.63 seconds

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501708, upper bound: 1.0502194
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0474348, upper bound: 1.0467715
time: 0.41 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1981437, 0.6735660, -0.2631326, 0.7698788, -0.9680225, 0.9366986
1: -0.4075772, 0.8701689, -0.5045100, 0.9689885, -1.3765657, 1.3746790
2: -0.3290880, 0.9651487, -0.4292829, 1.0956014, -1.4246894, 1.3944316
3: -0.7565911, 0.9793513, -0.8731403, 1.1296451, -1.8862362, 1.8524916
4: -0.6072877, 1.1647243, -0.7632787, 1.3396218, -1.9469094, 1.9280031

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 4

Time for candidate selection: 2.66 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0535787
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0535787
time: 0.44 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.2620909, 0.7696464, -0.2631326, 0.7698788, -1.0319697, 1.0327790
1: -0.5045343, 0.9919055, -0.5045100, 0.9689885, -1.4735228, 1.4964156
2: -0.4200620, 1.0960678, -0.4292829, 1.0956014, -1.5156634, 1.5253507
3: -0.8876588, 1.1395187, -0.8731403, 1.1296451, -2.0173039, 2.0126591
4: -0.7470124, 1.3261604, -0.7632787, 1.3396218, -2.0866342, 2.0894392

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 4

Time for candidate selection: 2.66 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0537102
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0537102
time: 0.37 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1981437, 0.6735660, -0.2254990, 0.7289720, -0.9271157, 0.8990650
1: -0.4075772, 0.8701689, -0.4573010, 0.9075529, -1.3151300, 1.3274699
2: -0.3290880, 0.9651487, -0.3811812, 1.0387152, -1.3678032, 1.3463299
3: -0.7565911, 0.9793513, -0.8068940, 1.0590353, -1.8156264, 1.7862453
4: -0.6072877, 1.1647243, -0.6912529, 1.2759079, -1.8831956, 1.8559773

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 4

Time for candidate selection: 2.80 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0534783, upper bound: 1.0535787
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0534783, upper bound: 1.0535787
time: 0.38 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.2620909, 0.7696464, -0.2254990, 0.7289720, -0.9910628, 0.9951454
1: -0.5045343, 0.9919055, -0.4573010, 0.9075529, -1.4120872, 1.4492066
2: -0.4200620, 1.0960678, -0.3811812, 1.0387152, -1.4587772, 1.4772490
3: -0.8876588, 1.1395187, -0.8068940, 1.0590353, -1.9466941, 1.9464128
4: -0.7470124, 1.3261604, -0.6912529, 1.2759079, -2.0229201, 2.0174134

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 4

Time for candidate selection: 2.75 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0534783, upper bound: 1.0537102
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0534783, upper bound: 1.0537102
time: 0.37 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1720736, 0.6441831, -0.2631326, 0.7698788, -0.9419523, 0.9073157
1: -0.3784128, 0.8175137, -0.5045100, 0.9689885, -1.3474014, 1.3220237
2: -0.3015163, 0.9219700, -0.4292829, 1.0956014, -1.3971177, 1.3512529
3: -0.7047867, 0.9291709, -0.8731403, 1.1296451, -1.8344318, 1.8023112
4: -0.5681181, 1.1211691, -0.7632787, 1.3396218, -1.9077399, 1.8844478

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0515558, upper bound: 1.0516661
time: 0.49 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 4

Time for candidate selection: 3.23 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0535655
time: 0.45 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0535655
time: 0.44 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.2310074, 0.7175490, -0.2631326, 0.7698788, -1.0008862, 0.9806816
1: -0.4606035, 0.9202803, -0.5045100, 0.9689885, -1.4295919, 1.4247904
2: -0.3788169, 1.0198433, -0.4292829, 1.0956014, -1.4744184, 1.4491262
3: -0.8183979, 1.0579665, -0.8731403, 1.1296451, -1.9480430, 1.9311068
4: -0.6781206, 1.2412384, -0.7632787, 1.3396218, -2.0177424, 2.0045171

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0516703, upper bound: 1.0534546
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0518665, upper bound: 1.0524409
time: 0.48 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 4

Time for candidate selection: 3.53 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0537102
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0537102
time: 0.43 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1720736, 0.6441831, -0.2254990, 0.7289720, -0.9010456, 0.8696821
1: -0.3784128, 0.8175137, -0.4573010, 0.9075529, -1.2859657, 1.2748147
2: -0.3015163, 0.9219700, -0.3811812, 1.0387152, -1.3402315, 1.3031512
3: -0.7047867, 0.9291709, -0.8068940, 1.0590353, -1.7638220, 1.7360649
4: -0.5681181, 1.1211691, -0.6912529, 1.2759079, -1.8440260, 1.8124220

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0515558, upper bound: 1.0516661
time: 0.46 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0514110, upper bound: 1.0509469
time: 0.51 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.2310074, 0.7175490, -0.2254990, 0.7289720, -0.9599794, 0.9430480
1: -0.4606035, 0.9202803, -0.4573010, 0.9075529, -1.3681564, 1.3775814
2: -0.3788169, 1.0198433, -0.3811812, 1.0387152, -1.4175322, 1.4010246
3: -0.8183979, 1.0579665, -0.8068940, 1.0590353, -1.8774332, 1.8648605
4: -0.6781206, 1.2412384, -0.6912529, 1.2759079, -1.9540285, 1.9324913

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0516703, upper bound: 1.0534546
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512376, upper bound: 1.0513465
time: 0.45 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.2060791, 0.7059609, -1.2135592, 2.4460342, -2.6521134, 1.9195201
1: -0.4219130, 0.8932547, -1.6813219, 2.6711469, -3.0930600, 2.5745766
2: -0.3510778, 1.0066929, -1.6465163, 3.0081160, -3.3591938, 2.6532092
3: -0.7678787, 1.0128576, -2.1652064, 3.4595845, -4.2274632, 3.1780639
4: -0.6492636, 1.2104864, -2.5053558, 3.5191135, -4.1683769, 3.7158422

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0525863, upper bound: 1.0490617
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0525863, upper bound: 1.0490617
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1713181, 0.6619900, -1.2136881, 2.4463463, -2.6176643, 1.8756781
1: -0.3777039, 0.8330576, -1.6815057, 2.6714954, -3.0491993, 2.5145633
2: -0.3047699, 0.9397614, -1.6466799, 3.0085113, -3.3132811, 2.5864413
3: -0.7021515, 0.9421769, -2.1654463, 3.4600346, -4.1621857, 3.1076231
4: -0.5747969, 1.1405221, -2.5056057, 3.5195577, -4.0943546, 3.6461277

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0525863, upper bound: 1.0490617
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0525863, upper bound: 1.0507390
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.2060791, 0.7059609, -1.1446891, 2.3593168, -2.5653958, 1.8506501
1: -0.4219130, 0.8932547, -1.5841074, 2.6174879, -3.0394008, 2.4773622
2: -0.3510778, 1.0066929, -1.5293760, 2.9587402, -3.3098180, 2.5360689
3: -0.7678787, 1.0128576, -2.1009860, 3.3199637, -4.0878425, 3.1138434
4: -0.6492636, 1.2104864, -2.3556986, 3.4134674, -4.0627308, 3.5661850

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0543227, upper bound: 1.0504137
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0540514, upper bound: 1.0491135
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1713181, 0.6619900, -1.1447992, 2.3595920, -2.5309100, 1.8067892
1: -0.3777039, 0.8330576, -1.5842650, 2.6178081, -2.9955120, 2.4173226
2: -0.3047699, 0.9397614, -1.5295095, 2.9590945, -3.2638645, 2.4692709
3: -0.7021515, 0.9421769, -2.1012015, 3.3203290, -4.0224800, 3.0433784
4: -0.5747969, 1.1405221, -2.3559012, 3.4138608, -3.9886577, 3.4964232

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0487347, upper bound: 1.0399864
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0544207, upper bound: 1.0504076
time: 0.45 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0544435, upper bound: 1.0491131
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1829112, 0.7200209, -1.2136881, 2.4463463, -2.6292572, 1.9337090
1: -0.3864989, 0.9190384, -1.6815057, 2.6714954, -3.0579944, 2.6005440
2: -0.3115277, 1.0421524, -1.6466799, 3.0085113, -3.3200390, 2.6888323
3: -0.7443157, 1.0138530, -2.1654463, 3.4600346, -4.2043505, 3.1792994
4: -0.6034802, 1.2099588, -2.5056057, 3.5195577, -4.1230378, 3.7155645

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0494371, upper bound: 1.0485268
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0494371, upper bound: 1.0494973
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.2494289, 0.7944719, -1.2136881, 2.4463463, -2.6957753, 2.0081601
1: -0.4813044, 1.0119214, -1.6815057, 2.6714954, -3.1527998, 2.6934271
2: -0.4077803, 1.1496273, -1.6466799, 3.0085113, -3.4162912, 2.7963071
3: -0.8642187, 1.1482675, -2.1654463, 3.4600346, -4.3242531, 3.3137138
4: -0.7469358, 1.3539220, -2.5056057, 3.5195577, -4.2664933, 3.8595276

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0494371, upper bound: 1.0485268
time: 0.45 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0494371, upper bound: 1.0494973
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1829112, 0.7200209, -1.1447992, 2.3595920, -2.5425029, 1.8648201
1: -0.3864989, 0.9190384, -1.5842650, 2.6178081, -3.0043070, 2.5033035
2: -0.3115277, 1.0421524, -1.5295095, 2.9590945, -3.2706223, 2.5716619
3: -0.7443157, 1.0138530, -2.1012015, 3.3203290, -4.0646448, 3.1150546
4: -0.6034802, 1.2099588, -2.3559012, 3.4138608, -4.0173407, 3.5658600

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502910, upper bound: 1.0488995
time: 0.46 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0498452, upper bound: 1.0482416
time: 0.47 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.2494289, 0.7944719, -1.1447992, 2.3595920, -2.6090207, 1.9392711
1: -0.4813044, 1.0119214, -1.5842650, 2.6178081, -3.0991123, 2.5961864
2: -0.4077803, 1.1496273, -1.5295095, 2.9590945, -3.3668747, 2.6791368
3: -0.8642187, 1.1482675, -2.1012015, 3.3203290, -4.1845474, 3.2494690
4: -0.7469358, 1.3539220, -2.3559012, 3.4138608, -4.1607962, 3.7098231

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502910, upper bound: 1.0488995
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0498452, upper bound: 1.0482416
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1981437, 0.6735660, -1.2317941, 2.5485647, -2.7467084, 1.9053602
1: -0.4075772, 0.8701689, -1.7097392, 2.7696524, -3.1772294, 2.5799081
2: -0.3290880, 0.9651487, -1.6688566, 3.1283512, -3.4574392, 2.6340053
3: -0.7565911, 0.9793513, -2.2169631, 3.5870750, -4.3436656, 3.1963143
4: -0.6072877, 1.1647243, -2.5448000, 3.6422343, -4.2495217, 3.7095244

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 4

Time for candidate selection: 2.69 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507335, upper bound: 1.0491179
time: 0.44 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507335, upper bound: 1.0491179
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.2620909, 0.7696464, -1.2317941, 2.5485647, -2.8106556, 2.0014405
1: -0.5045343, 0.9919055, -1.7097392, 2.7696524, -3.2741861, 2.7016447
2: -0.4200620, 1.0960678, -1.6688566, 3.1283512, -3.5484128, 2.7649245
3: -0.8876588, 1.1395187, -2.2169631, 3.5870750, -4.4747338, 3.3564818
4: -0.7470124, 1.3261604, -2.5448000, 3.6422343, -4.3892469, 3.8709605

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 4

Time for candidate selection: 2.70 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507335, upper bound: 1.0492494
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507335, upper bound: 1.0492494
time: 0.45 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1981437, 0.6735660, -1.3011413, 2.5799091, -2.7780528, 1.9747074
1: -0.4075772, 0.8701689, -1.7600503, 2.8037138, -3.2112908, 2.6302192
2: -0.3290880, 0.9651487, -1.7155523, 3.1846526, -3.5137405, 2.6807010
3: -0.7565911, 0.9793513, -2.2817159, 3.6491742, -4.4057636, 3.2610672
4: -0.6072877, 1.1647243, -2.6121106, 3.7078054, -4.3150930, 3.7768350

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 4

Time for candidate selection: 2.67 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508041, upper bound: 1.0491440
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508041, upper bound: 1.0491440
time: 0.44 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.2620909, 0.7696464, -1.3011413, 2.5799091, -2.8420000, 2.0707877
1: -0.5045343, 0.9919055, -1.7600503, 2.8037138, -3.3082476, 2.7519557
2: -0.4200620, 1.0960678, -1.7155523, 3.1846526, -3.6047144, 2.8116202
3: -0.8876588, 1.1395187, -2.2817159, 3.6491742, -4.5368319, 3.4212346
4: -0.7470124, 1.3261604, -2.6121106, 3.7078054, -4.4548173, 3.9382710

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 4

Time for candidate selection: 2.73 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508041, upper bound: 1.0492755
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508041, upper bound: 1.0492755
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1720736, 0.6441831, -1.2317941, 2.5485647, -2.7206383, 1.8759773
1: -0.3784128, 0.8175137, -1.7097392, 2.7696524, -3.1480651, 2.5272529
2: -0.3015163, 0.9219700, -1.6688566, 3.1283512, -3.4298673, 2.5908265
3: -0.7047867, 0.9291709, -2.2169631, 3.5870750, -4.2918615, 3.1461339
4: -0.5681181, 1.1211691, -2.5448000, 3.6422343, -4.2103524, 3.6659691

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0487461, upper bound: 1.0472668
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 4

Time for candidate selection: 3.25 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507335, upper bound: 1.0491047
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507335, upper bound: 1.0491047
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.2310074, 0.7175490, -1.2317941, 2.5485647, -2.7795720, 1.9493432
1: -0.4606035, 0.9202803, -1.7097392, 2.7696524, -3.2302556, 2.6300194
2: -0.3788169, 1.0198433, -1.6688566, 3.1283512, -3.5071678, 2.6887000
3: -0.8183979, 1.0579665, -2.2169631, 3.5870750, -4.4054728, 3.2749295
4: -0.6781206, 1.2412384, -2.5448000, 3.6422343, -4.3203549, 3.7860384

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0490569, upper bound: 1.0480416
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0481298, upper bound: 1.0479691
time: 0.46 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1720736, 0.6441831, -1.3011413, 2.5799091, -2.7519822, 1.9453244
1: -0.3784128, 0.8175137, -1.7600503, 2.8037138, -3.1821265, 2.5775640
2: -0.3015163, 0.9219700, -1.7155523, 3.1846526, -3.4861686, 2.6375222
3: -0.7047867, 0.9291709, -2.2817159, 3.6491742, -4.3539600, 3.2108867
4: -0.5681181, 1.1211691, -2.6121106, 3.7078054, -4.2759228, 3.7332797

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0487461, upper bound: 1.0472668
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 4

Time for candidate selection: 3.16 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507335, upper bound: 1.0491308
time: 0.44 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507335, upper bound: 1.0491308
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.2310074, 0.7175490, -1.3011413, 2.5799091, -2.8109164, 2.0186903
1: -0.4606035, 0.9202803, -1.7600503, 2.8037138, -3.2643173, 2.6803305
2: -0.3788169, 1.0198433, -1.7155523, 3.1846526, -3.5634689, 2.7353957
3: -0.8183979, 1.0579665, -2.2817159, 3.6491742, -4.4675713, 3.3396823
4: -0.6781206, 1.2412384, -2.6121106, 3.7078054, -4.3859262, 3.8533490

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0490569, upper bound: 1.0481479
time: 0.44 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 4

Time for candidate selection: 3.18 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507335, upper bound: 1.0492755
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507335, upper bound: 1.0492755
time: 0.50 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1981437, 0.6735660, -1.2890360, 2.6080267, -2.8061705, 1.9626021
1: -0.4075772, 0.8701689, -1.7757368, 2.8224382, -3.2300153, 2.6459057
2: -0.3290880, 0.9651487, -1.7412896, 3.1970921, -3.5261800, 2.7064383
3: -0.7565911, 0.9793513, -2.2828507, 3.6829684, -4.4395590, 3.2622020
4: -0.6072877, 1.1647243, -2.6450953, 3.7347975, -4.3420854, 3.8098197

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 6

Time for candidate selection: 2.69 seconds

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507640, upper bound: 1.0493151
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507640, upper bound: 1.0495970
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.2620909, 0.7696464, -1.2890360, 2.6080267, -2.8701177, 2.0586824
1: -0.5045343, 0.9919055, -1.7757368, 2.8224382, -3.3269725, 2.7676423
2: -0.4200620, 1.0960678, -1.7412896, 3.1970921, -3.6171536, 2.8373575
3: -0.8876588, 1.1395187, -2.2828507, 3.6829684, -4.5706272, 3.4223695
4: -0.7470124, 1.3261604, -2.6450953, 3.7347975, -4.4818096, 3.9712558

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 6

Time for candidate selection: 2.73 seconds

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507640, upper bound: 1.0494466
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507640, upper bound: 1.0497285
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1981437, 0.6735660, -1.3504868, 2.6251640, -2.8233078, 2.0240529
1: -0.4075772, 0.8701689, -1.8179705, 2.8394029, -3.2469802, 2.6881394
2: -0.3290880, 0.9651487, -1.7806034, 3.2355962, -3.5646842, 2.7457521
3: -0.7565911, 0.9793513, -2.3383503, 3.7266910, -4.4832807, 3.3177016
4: -0.6072877, 1.1647243, -2.7018437, 3.7814958, -4.3887835, 3.8665681

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 4

Time for candidate selection: 2.71 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508706, upper bound: 1.0495603
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508706, upper bound: 1.0498604
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.2620909, 0.7696464, -1.3504868, 2.6251640, -2.8872550, 2.1201332
1: -0.5045343, 0.9919055, -1.8179705, 2.8394029, -3.3439369, 2.8098760
2: -0.4200620, 1.0960678, -1.7806034, 3.2355962, -3.6556582, 2.8766713
3: -0.8876588, 1.1395187, -2.3383503, 3.7266910, -4.6143489, 3.4778690
4: -0.7470124, 1.3261604, -2.7018437, 3.7814958, -4.5285082, 4.0280042

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 4

Time for candidate selection: 2.73 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508706, upper bound: 1.0496703
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508706, upper bound: 1.0499680
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.2013567, 0.6970940, -1.3504868, 2.6251640, -2.8265207, 2.0475807
1: -0.4231896, 0.8673437, -1.8179705, 2.8394029, -3.2625926, 2.6853142
2: -0.3478338, 0.9940906, -1.7806034, 3.2355962, -3.5834298, 2.7746940
3: -0.7603652, 1.0071222, -2.3383503, 3.7266910, -4.4870558, 3.3454723
4: -0.6426137, 1.2188928, -2.7018437, 3.7814958, -4.4241095, 3.9207366

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0498321, upper bound: 1.0506330
time: 0.44 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0498323, upper bound: 1.0506330
time: 0.45 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.2909008, 0.8438774, -1.3320332, 2.5860474, -2.8769479, 2.1759105
1: -0.5446191, 1.0634959, -1.7931142, 2.7968993, -3.3415182, 2.8566101
2: -0.4665691, 1.2150321, -1.7571807, 3.1905499, -3.6571190, 2.9722128
3: -0.9410769, 1.2407138, -2.3080869, 3.6721001, -4.6131768, 3.5488007
4: -0.8382176, 1.4520541, -2.6684370, 3.7299933, -4.5682106, 4.1204910

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512131, upper bound: 1.0506391
time: 0.51 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512131, upper bound: 1.0506391
time: 0.49 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 4.22 seconds
IS_A1_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0555443, upper bound: 1.0555443
IS_A1_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0555443, upper bound: 1.0555443
IS_A1_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0555443, upper bound: 1.0555617
IS_A1_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0555443, upper bound: 1.0555617
IS_A1_B1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0537774, upper bound: 1.0509606
IS_A1_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0533379, upper bound: 1.0508971
IS_A1_B1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0550041, upper bound: 1.0510182
IS_A1_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0548845, upper bound: 1.0509748
IS_A1_B1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0509606, upper bound: 1.0537774
IS_A1_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0510182, upper bound: 1.0550041
IS_A1_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0508971, upper bound: 1.0533379
IS_A1_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0509748, upper bound: 1.0548845
IS_A1_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0501055, upper bound: 1.0501055
IS_A1_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0501055, upper bound: 1.0501055
IS_A1_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0501055, upper bound: 1.0501055
IS_A1_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0501055, upper bound: 1.0501055
IS_A1_B1_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0554253, upper bound: 1.0539330
IS_A1_B1_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0554253, upper bound: 1.0539330
IS_A1_B1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0554253, upper bound: 1.0539330
IS_A1_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0554253, upper bound: 1.0539330
IS_A1_B1_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0545562, upper bound: 1.0511174
IS_A1_B1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0544140, upper bound: 1.0508131
IS_A1_B1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0549649, upper bound: 1.0511174
IS_A1_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0550041, upper bound: 1.0508972
IS_A1_B1_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0499481, upper bound: 1.0504579
IS_A1_B1_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0502251, upper bound: 1.0509033
IS_A1_B1_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0499013, upper bound: 1.0497225
IS_A1_B1_A1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0494542, upper bound: 1.0483698
IS_A1_B1_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0550223
IS_A1_B1_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0550282
IS_A1_B1_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0551914
IS_A1_B1_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0551945
IS_A1_B1_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0534783, upper bound: 1.0549755
IS_A1_B1_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0534783, upper bound: 1.0550006
IS_A1_B1_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0534783, upper bound: 1.0551410
IS_A1_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0534783, upper bound: 1.0551583
IS_A1_B1_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0518924, upper bound: 1.0551300
IS_A1_B1_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0518924, upper bound: 1.0552379
IS_A1_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0518924, upper bound: 1.0551300
IS_A1_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0518924, upper bound: 1.0552379
IS_A1_B1_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0501708, upper bound: 1.0502084
IS_A1_B1_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0474348, upper bound: 1.0469382
IS_A1_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0501708, upper bound: 1.0502194
IS_A1_B1_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0474348, upper bound: 1.0467715
IS_A1_B1_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0535787
IS_A1_B1_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0535787
IS_A1_B1_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0537102
IS_A1_B1_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0537102
IS_A1_B1_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0534783, upper bound: 1.0535787
IS_A1_B1_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0534783, upper bound: 1.0535787
IS_A1_B1_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0534783, upper bound: 1.0537102
IS_A1_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0534783, upper bound: 1.0537102
IS_A1_B1_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0535655
IS_A1_B1_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0535655
IS_A1_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0537102
IS_A1_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0537102
IS_A1_B1_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0515558, upper bound: 1.0516661
IS_A1_B1_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0514110, upper bound: 1.0509469
IS_A1_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0516703, upper bound: 1.0534546
IS_A1_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0512376, upper bound: 1.0513465
IS_A1_B2_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0525863, upper bound: 1.0490617
IS_A1_B2_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0525863, upper bound: 1.0490617
IS_A1_B2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0525863, upper bound: 1.0490617
IS_A1_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0525863, upper bound: 1.0507390
IS_A1_B2_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0543227, upper bound: 1.0504137
IS_A1_B2_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0540514, upper bound: 1.0491135
IS_A1_B2_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0544207, upper bound: 1.0504076
IS_A1_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0544435, upper bound: 1.0491131
IS_A1_B2_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0494371, upper bound: 1.0485268
IS_A1_B2_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0494371, upper bound: 1.0494973
IS_A1_B2_A1_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0494371, upper bound: 1.0485268
IS_A1_B2_A1_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0494371, upper bound: 1.0494973
IS_A1_B2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0502910, upper bound: 1.0488995
IS_A1_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0498452, upper bound: 1.0482416
IS_A1_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0502910, upper bound: 1.0488995
IS_A1_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0498452, upper bound: 1.0482416
IS_A1_B2_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0507335, upper bound: 1.0491179
IS_A1_B2_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0507335, upper bound: 1.0491179
IS_A1_B2_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0507335, upper bound: 1.0492494
IS_A1_B2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0507335, upper bound: 1.0492494
IS_A1_B2_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0508041, upper bound: 1.0491440
IS_A1_B2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0508041, upper bound: 1.0491440
IS_A1_B2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0508041, upper bound: 1.0492755
IS_A1_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0508041, upper bound: 1.0492755
IS_A1_B2_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0507335, upper bound: 1.0491047
IS_A1_B2_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0507335, upper bound: 1.0491047
IS_A1_B2_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0490569, upper bound: 1.0480416
IS_A1_B2_A2_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0481298, upper bound: 1.0479691
IS_A1_B2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0507335, upper bound: 1.0491308
IS_A1_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0507335, upper bound: 1.0491308
IS_A1_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0507335, upper bound: 1.0492755
IS_A1_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0507335, upper bound: 1.0492755
IS_A1_B2_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0507640, upper bound: 1.0493151
IS_A1_B2_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0507640, upper bound: 1.0495970
IS_A1_B2_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0507640, upper bound: 1.0494466
IS_A1_B2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0507640, upper bound: 1.0497285
IS_A1_B2_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0508706, upper bound: 1.0495603
IS_A1_B2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0508706, upper bound: 1.0498604
IS_A1_B2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0508706, upper bound: 1.0496703
IS_A1_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0508706, upper bound: 1.0499680
IS_A1_B2_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0498321, upper bound: 1.0506330
IS_A1_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0498323, upper bound: 1.0506330
IS_A1_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0512131, upper bound: 1.0506391
IS_A1_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.22
Output dim: 0, lower bound: -1.0512131, upper bound: 1.0506391

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.2096419, 0.7101620, -0.2096419, 0.7101620, -0.9198039, 0.9198039
1: -0.4267260, 0.8983275, -0.4267260, 0.8983275, -1.3250535, 1.3250535
2: -0.3558276, 1.0129061, -0.3558276, 1.0129061, -1.3687336, 1.3687336
3: -0.7740620, 1.0201761, -0.7740620, 1.0201761, -1.7942381, 1.7942381
4: -0.6563050, 1.2186592, -0.6563050, 1.2186592, -1.8749641, 1.8749641

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0529589, upper bound: 1.0550976
time: 0.47 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0553212, upper bound: 1.0553205
time: 0.40 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 2.77 seconds
IS_A1_B1_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.77
Output dim: 0, lower bound: -1.0529589, upper bound: 1.0550976
IS_A1_B1_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.77
Output dim: 0, lower bound: -1.0553212, upper bound: 1.0553205
IS_A1_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0555443, upper bound: 1.0555443
IS_A1_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0555443, upper bound: 1.0555617
IS_A1_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0555443, upper bound: 1.0555617
IS_A1_B1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0537774, upper bound: 1.0509606
IS_A1_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0533379, upper bound: 1.0508971
IS_A1_B1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0550041, upper bound: 1.0510182
IS_A1_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0548845, upper bound: 1.0509748
IS_A1_B1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0509606, upper bound: 1.0537774
IS_A1_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0510182, upper bound: 1.0550041
IS_A1_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0508971, upper bound: 1.0533379
IS_A1_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0509748, upper bound: 1.0548845
IS_A1_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0501055, upper bound: 1.0501055
IS_A1_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0501055, upper bound: 1.0501055
IS_A1_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0501055, upper bound: 1.0501055
IS_A1_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0501055, upper bound: 1.0501055
IS_A1_B1_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0554253, upper bound: 1.0539330
IS_A1_B1_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0554253, upper bound: 1.0539330
IS_A1_B1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0554253, upper bound: 1.0539330
IS_A1_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0554253, upper bound: 1.0539330
IS_A1_B1_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0545562, upper bound: 1.0511174
IS_A1_B1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0544140, upper bound: 1.0508131
IS_A1_B1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0549649, upper bound: 1.0511174
IS_A1_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0550041, upper bound: 1.0508972
IS_A1_B1_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0499481, upper bound: 1.0504579
IS_A1_B1_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0502251, upper bound: 1.0509033
IS_A1_B1_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0499013, upper bound: 1.0497225
IS_A1_B1_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0550223
IS_A1_B1_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0550282
IS_A1_B1_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0551914
IS_A1_B1_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0551945
IS_A1_B1_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0534783, upper bound: 1.0549755
IS_A1_B1_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0534783, upper bound: 1.0550006
IS_A1_B1_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0534783, upper bound: 1.0551410
IS_A1_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0534783, upper bound: 1.0551583
IS_A1_B1_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0518924, upper bound: 1.0551300
IS_A1_B1_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0518924, upper bound: 1.0552379
IS_A1_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0518924, upper bound: 1.0551300
IS_A1_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0518924, upper bound: 1.0552379
IS_A1_B1_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0501708, upper bound: 1.0502084
IS_A1_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0501708, upper bound: 1.0502194
IS_A1_B1_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0535787
IS_A1_B1_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0535787
IS_A1_B1_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0537102
IS_A1_B1_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0537102
IS_A1_B1_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0534783, upper bound: 1.0535787
IS_A1_B1_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0534783, upper bound: 1.0535787
IS_A1_B1_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0534783, upper bound: 1.0537102
IS_A1_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0534783, upper bound: 1.0537102
IS_A1_B1_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0535655
IS_A1_B1_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0535655
IS_A1_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0537102
IS_A1_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0537102
IS_A1_B1_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0515558, upper bound: 1.0516661
IS_A1_B1_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0514110, upper bound: 1.0509469
IS_A1_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0516703, upper bound: 1.0534546
IS_A1_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0512376, upper bound: 1.0513465
IS_A1_B2_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0525863, upper bound: 1.0490617
IS_A1_B2_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0525863, upper bound: 1.0490617
IS_A1_B2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0525863, upper bound: 1.0490617
IS_A1_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0525863, upper bound: 1.0507390
IS_A1_B2_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0543227, upper bound: 1.0504137
IS_A1_B2_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0540514, upper bound: 1.0491135
IS_A1_B2_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0544207, upper bound: 1.0504076
IS_A1_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0544435, upper bound: 1.0491131
IS_A1_B2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0502910, upper bound: 1.0488995
IS_A1_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0498452, upper bound: 1.0482416
IS_A1_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0502910, upper bound: 1.0488995
IS_A1_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0498452, upper bound: 1.0482416
IS_A1_B2_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0507335, upper bound: 1.0491179
IS_A1_B2_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0507335, upper bound: 1.0491179
IS_A1_B2_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0507335, upper bound: 1.0492494
IS_A1_B2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0507335, upper bound: 1.0492494
IS_A1_B2_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0508041, upper bound: 1.0491440
IS_A1_B2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0508041, upper bound: 1.0491440
IS_A1_B2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0508041, upper bound: 1.0492755
IS_A1_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0508041, upper bound: 1.0492755
IS_A1_B2_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0507335, upper bound: 1.0491047
IS_A1_B2_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0507335, upper bound: 1.0491047
IS_A1_B2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0507335, upper bound: 1.0491308
IS_A1_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0507335, upper bound: 1.0491308
IS_A1_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0507335, upper bound: 1.0492755
IS_A1_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0507335, upper bound: 1.0492755
IS_A1_B2_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0507640, upper bound: 1.0493151
IS_A1_B2_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0507640, upper bound: 1.0495970
IS_A1_B2_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0507640, upper bound: 1.0494466
IS_A1_B2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0507640, upper bound: 1.0497285
IS_A1_B2_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0508706, upper bound: 1.0495603
IS_A1_B2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0508706, upper bound: 1.0498604
IS_A1_B2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0508706, upper bound: 1.0496703
IS_A1_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0508706, upper bound: 1.0499680
IS_A1_B2_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0498321, upper bound: 1.0506330
IS_A1_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0498323, upper bound: 1.0506330
IS_A1_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0512131, upper bound: 1.0506391
IS_A1_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.77
Output dim: 0, lower bound: -1.0512131, upper bound: 1.0506391
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.1000000, mid=0.1000000, abs_max=1.1883488893508911
rel_dist={0: [-1.0558835892060525, 1.0558835892060525]}

## Binary search (step 1) starts
Candidate diff: 0.0500000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0540802, upper bound: 1.0510815
time: 0.38 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
time: 0.34 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.88 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.88
Output dim: 0, lower bound: -1.0540802, upper bound: 1.0510815
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.88
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.2352601, 0.7538407, -0.3035725, 0.8847764, -1.1200366, 1.0574131
1: -0.4706453, 0.9488738, -0.5660125, 1.0933844, -1.5640295, 1.5148864
2: -0.3915833, 1.0723588, -0.4826685, 1.2412479, -1.6328310, 1.5550274
3: -0.8320177, 1.0878556, -0.9617165, 1.2755736, -2.1075912, 2.0495720
4: -0.7029035, 1.3031529, -0.8354526, 1.4994075, -2.2023110, 2.1386056

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
time: 0.36 seconds

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

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
time: 0.37 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
time: 0.37 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.35 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.35
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.35
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.35
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.35
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.2352601, 0.7538407, -0.2352601, 0.7538407, -0.9891008, 0.9891006
1: -0.4706453, 0.9488738, -0.4706453, 0.9488738, -1.4195192, 1.4195192
2: -0.3915833, 1.0723588, -0.3915833, 1.0723588, -1.4639422, 1.4639422
3: -0.8320177, 1.0878556, -0.8320177, 1.0878556, -1.9198732, 1.9198732
4: -0.7029035, 1.3031529, -0.7029035, 1.3031529, -2.0060563, 2.0060563

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535217, upper bound: 1.0507277
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0531956, upper bound: 1.0509582
time: 0.35 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.2352601, 0.7538407, -1.3271873, 2.7002649, -2.9355247, 2.0810280
1: -0.4706453, 0.9488738, -1.8266034, 2.9257355, -3.3963809, 2.7754772
2: -0.3915833, 1.0723588, -1.7866864, 3.3100519, -3.7016354, 2.8590453
3: -0.8320177, 1.0878556, -2.3538351, 3.8103127, -4.6423302, 3.4416907
4: -0.7029035, 1.3031529, -2.7129741, 3.8588223, -4.5617256, 4.0161266

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535217, upper bound: 1.0507277
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0531956, upper bound: 1.0509582
time: 0.38 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -1.3271873, 2.7002649, -0.2352601, 0.7538407, -2.0810280, 2.9355249
1: -1.8266034, 2.9257355, -0.4706453, 0.9488738, -2.7754772, 3.3963809
2: -1.7866864, 3.3100519, -0.3915833, 1.0723588, -2.8590453, 3.7016351
3: -2.3538351, 3.8103127, -0.8320177, 1.0878556, -3.4416907, 4.6423302
4: -2.7129741, 3.8588223, -0.7029035, 1.3031529, -4.0161266, 4.5617256

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0484906, upper bound: 1.0491638
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0477990, upper bound: 1.0477990
time: 0.35 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -1.3271873, 2.7002649, -1.3271873, 2.7002649, -4.0274525, 4.0274525
1: -1.8266034, 2.9257355, -1.8266034, 2.9257355, -4.7523389, 4.7523384
2: -1.7866864, 3.3100519, -1.7866864, 3.3100519, -5.0967383, 5.0967379
3: -2.3538351, 3.8103127, -2.3538351, 3.8103127, -6.1641479, 6.1641479
4: -2.7129741, 3.8588223, -2.7129741, 3.8588223, -6.5717964, 6.5717964

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0484906, upper bound: 1.0491638
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0477990, upper bound: 1.0477990
time: 0.35 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.33 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 0, lower bound: -1.0535217, upper bound: 1.0507277
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 0, lower bound: -1.0531956, upper bound: 1.0509582
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 0, lower bound: -1.0535217, upper bound: 1.0507277
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 0, lower bound: -1.0531956, upper bound: 1.0509582
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 2.33
Output dim: 0, lower bound: -1.0484906, upper bound: 1.0491638
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.33
Output dim: 0, lower bound: -1.0477990, upper bound: 1.0477990
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 2.33
Output dim: 0, lower bound: -1.0484906, upper bound: 1.0491638
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.33
Output dim: 0, lower bound: -1.0477990, upper bound: 1.0477990

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.2013956, 0.7023641, -0.2304998, 0.7457045, -0.9471001, 0.9328639
1: -0.4206367, 0.8831170, -0.4635985, 0.9387211, -1.3593577, 1.3467155
2: -0.3466128, 1.0003821, -0.3852943, 1.0613912, -1.4080040, 1.3856764
3: -0.7602279, 1.0080743, -0.8218359, 1.0758421, -1.8360701, 1.8299102
4: -0.6370696, 1.2147777, -0.6937719, 1.2902606, -1.9273301, 1.9085495

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539857, upper bound: 1.0539857
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539857, upper bound: 1.0539857
time: 0.35 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.2363607, 0.7416377, -0.2258822, 0.7346829, -0.9710436, 0.9675199
1: -0.4722191, 0.9222932, -0.4567959, 0.9236316, -1.3958507, 1.3790891
2: -0.3964722, 1.0577438, -0.3797194, 1.0467503, -1.4432225, 1.4374632
3: -0.8244337, 1.0810699, -0.8123593, 1.0611860, -1.8856196, 1.8934293
4: -0.7141775, 1.3009543, -0.6861970, 1.2757928, -1.9899704, 1.9871514

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

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
0: -0.2013956, 0.7023641, -1.3238299, 2.6921263, -2.8935215, 2.0261941
1: -0.4206367, 0.8831170, -1.8218448, 2.9161830, -3.3368196, 2.7049618
2: -0.3466128, 1.0003821, -1.7826023, 3.2999420, -3.6465545, 2.7829843
3: -0.7602279, 1.0080743, -2.3470442, 3.7989707, -4.5591974, 3.3551185
4: -0.6370696, 1.2147777, -2.7069280, 3.8476067, -4.4846764, 3.9217057

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535217, upper bound: 1.0502633
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0475518, upper bound: 1.0437452
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.2363607, 0.7416377, -1.3112228, 2.6565771, -2.8929377, 2.0528605
1: -0.4722191, 0.9222932, -1.8038421, 2.8760159, -3.3482347, 2.7261353
2: -0.3964722, 1.0577438, -1.7678263, 3.2591543, -3.6556265, 2.8255701
3: -0.8244337, 1.0810699, -2.3217871, 3.7515779, -4.5760117, 3.4028571
4: -0.7141775, 1.3009543, -2.6856859, 3.8030729, -4.5172505, 3.9866402

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513845, upper bound: 1.0496162
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0531956, upper bound: 1.0509582
time: 0.42 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.89 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 0, lower bound: -1.0539857, upper bound: 1.0539857
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 0, lower bound: -1.0539857, upper bound: 1.0539857
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 0, lower bound: -1.0539857, upper bound: 1.0540170
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 0, lower bound: -1.0539857, upper bound: 1.0540170
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 0, lower bound: -1.0535217, upper bound: 1.0502633
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.89
Output dim: 0, lower bound: -1.0475518, upper bound: 1.0437452
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 0, lower bound: -1.0513845, upper bound: 1.0496162
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 0, lower bound: -1.0531956, upper bound: 1.0509582

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.2013956, 0.7023641, -0.2013956, 0.7023641, -0.9037597, 0.9037597
1: -0.4206367, 0.8831170, -0.4206367, 0.8831170, -1.3037536, 1.3037536
2: -0.3466128, 1.0003821, -0.3466128, 1.0003821, -1.3469949, 1.3469949
3: -0.7602279, 1.0080743, -0.7602279, 1.0080743, -1.7683022, 1.7683022
4: -0.6370696, 1.2147777, -0.6370696, 1.2147777, -1.8518473, 1.8518473

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0533473, upper bound: 1.0508182
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502708, upper bound: 1.0502708
time: 0.44 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.2013956, 0.7023641, -0.2363607, 0.7416377, -0.9430333, 0.9387248
1: -0.4206367, 0.8831170, -0.4722191, 0.9222932, -1.3429298, 1.3553361
2: -0.3466128, 1.0003821, -0.3964722, 1.0577438, -1.4043566, 1.3968543
3: -0.7602279, 1.0080743, -0.8244337, 1.0810699, -1.8412979, 1.8325080
4: -0.6370696, 1.2147777, -0.7141775, 1.3009543, -1.9380239, 1.9289553

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0533473, upper bound: 1.0508182
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502708, upper bound: 1.0508893
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.2363607, 0.7416377, -0.2013956, 0.7023641, -0.9387248, 0.9430333
1: -0.4722191, 0.9222932, -0.4206367, 0.8831170, -1.3553361, 1.3429298
2: -0.3964722, 1.0577438, -0.3466128, 1.0003821, -1.3968543, 1.4043566
3: -0.8244337, 1.0810699, -0.7602279, 1.0080743, -1.8325080, 1.8412979
4: -0.7141775, 1.3009543, -0.6370696, 1.2147777, -1.9289553, 1.9380239

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.2363607, 0.7416377, -0.2363607, 0.7416377, -0.9779984, 0.9779984
1: -0.4722191, 0.9222932, -0.4722191, 0.9222932, -1.3945123, 1.3945123
2: -0.3964722, 1.0577438, -0.3964722, 1.0577438, -1.4542160, 1.4542160
3: -0.8244337, 1.0810699, -0.8244337, 1.0810699, -1.9055036, 1.9055036
4: -0.7141775, 1.3009543, -0.7141775, 1.3009543, -2.0151320, 2.0151320

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
time: 0.46 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1843704, 0.6816336, -1.2246854, 2.4703135, -2.6546841, 1.9063190
1: -0.3962561, 0.8586243, -1.6965811, 2.6998594, -3.0961154, 2.5552053
2: -0.3223111, 0.9689015, -1.6591611, 3.0394058, -3.3617170, 2.6280627
3: -0.7312128, 0.9729355, -2.1866965, 3.4946685, -4.2258811, 3.1596320
4: -0.6009668, 1.1744217, -2.5240049, 3.5537622, -4.1547289, 3.6984267

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0529566, upper bound: 1.0500381
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501373, upper bound: 1.0488327
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.2342383, 0.7380306, -1.2413032, 2.5691936, -2.8034320, 1.9793339
1: -0.4692839, 0.9176295, -1.7233796, 2.7961643, -3.2654481, 2.6410091
2: -0.3937626, 1.0519434, -1.6813955, 3.1583161, -3.5520785, 2.7333388
3: -0.8200877, 1.0757111, -2.2370095, 3.6173403, -4.4374275, 3.3127208
4: -0.7099589, 1.2945570, -2.5641105, 3.6751060, -4.3850651, 3.8586674

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513845, upper bound: 1.0496162
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513845, upper bound: 1.0496162
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.2353991, 0.7405176, -1.2977409, 2.6237350, -2.8591340, 2.0382586
1: -0.4708995, 0.9210051, -1.7878509, 2.8421307, -3.3130302, 2.7088561
2: -0.3951180, 1.0560383, -1.7525644, 3.2204282, -3.6155462, 2.8086028
3: -0.8228747, 1.0791245, -2.3005936, 3.7071474, -4.5300221, 3.3797181
4: -0.7121502, 1.2987292, -2.6624808, 3.7612123, -4.4733624, 3.9612100

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0529291, upper bound: 1.0507063
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0529291, upper bound: 1.0509582
time: 0.34 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.37 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -1.0533473, upper bound: 1.0508182
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -1.0502708, upper bound: 1.0502708
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -1.0533473, upper bound: 1.0508182
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -1.0502708, upper bound: 1.0508893
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -1.0529566, upper bound: 1.0500381
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -1.0501373, upper bound: 1.0488327
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -1.0513845, upper bound: 1.0496162
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -1.0513845, upper bound: 1.0496162
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -1.0529291, upper bound: 1.0507063
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -1.0529291, upper bound: 1.0509582

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1851257, 0.6783313, -0.2013956, 0.7023641, -0.8874898, 0.8797269
1: -0.3969364, 0.8526834, -0.4206367, 0.8831170, -1.2800534, 1.2733200
2: -0.3242711, 0.9644121, -0.3466128, 1.0003821, -1.3246531, 1.3110249
3: -0.7251614, 0.9704387, -0.7602279, 1.0080743, -1.7332357, 1.7306666
4: -0.6038694, 1.1718525, -0.6370696, 1.2147777, -1.8186471, 1.8089221

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502708, upper bound: 1.0502708
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502708, upper bound: 1.0502708
time: 0.43 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.2674521, 0.8277271, -0.1816620, 0.6706874, -0.9381396, 1.0093890
1: -0.5075184, 1.0548140, -0.3886601, 0.8448312, -1.3523496, 1.4434741
2: -0.4327361, 1.2001319, -0.3174605, 0.9567611, -1.3894973, 1.5175924
3: -0.9017408, 1.1985904, -0.7238576, 0.9572418, -1.8589826, 1.9224480
4: -0.7864016, 1.4086894, -0.5944541, 1.1581156, -1.9445173, 2.0031433

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502708, upper bound: 1.0502708
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502708, upper bound: 1.0502708
time: 0.37 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1851257, 0.6783313, -0.2363607, 0.7416377, -0.9267634, 0.9146920
1: -0.3969364, 0.8526834, -0.4722191, 0.9222932, -1.3192296, 1.3249025
2: -0.3242711, 0.9644121, -0.3964722, 1.0577438, -1.3820149, 1.3608843
3: -0.7251614, 0.9704387, -0.8244337, 1.0810699, -1.8062314, 1.7948724
4: -0.6038694, 1.1718525, -0.7141775, 1.3009543, -1.9048238, 1.8860300

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502708, upper bound: 1.0508892
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502708, upper bound: 1.0502708
time: 0.41 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.2674521, 0.8277271, -0.2196092, 0.7150588, -0.9825109, 1.0473363
1: -0.5075184, 1.0548140, -0.4448397, 0.8919341, -1.3994524, 1.4996537
2: -0.4327361, 1.2001319, -0.3713522, 1.0230851, -1.4558213, 1.5714841
3: -0.9017408, 1.1985904, -0.7955373, 1.0377979, -1.9395387, 1.9941278
4: -0.7864016, 1.4086894, -0.6779964, 1.2520282, -2.0384297, 2.0866857

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502708, upper bound: 1.0508892
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505944, upper bound: 1.0508893
time: 0.41 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.2631326, 0.7698788, -0.1990446, 0.6983105, -0.9614431, 0.9689233
1: -0.5045100, 0.9689885, -0.4173143, 0.8780323, -1.3825424, 1.3863027
2: -0.4292829, 1.0956014, -0.3434845, 0.9941077, -1.4233906, 1.4390860
3: -0.8731403, 1.1296451, -0.7553740, 1.0016891, -1.8748294, 1.8850191
4: -0.7632787, 1.3396218, -0.6323574, 1.2077138, -1.9709926, 1.9719791

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0548332
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0548332
time: 0.44 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.2254990, 0.7289720, -0.2004953, 0.7012957, -0.9267947, 0.9294673
1: -0.4573010, 0.9075529, -0.4193918, 0.8818332, -1.3391342, 1.3269446
2: -0.3811812, 1.0387152, -0.3453345, 0.9987528, -1.3799341, 1.3840497
3: -0.8068940, 1.0590353, -0.7587277, 1.0062188, -1.8131127, 1.8177630
4: -0.6912529, 1.2759079, -0.6351777, 1.2127244, -1.9039774, 1.9110856

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0514621, upper bound: 1.0534027
time: 0.46 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508892, upper bound: 1.0505944
time: 0.44 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.2631326, 0.7698788, -0.2342383, 0.7380306, -1.0011632, 1.0041171
1: -0.5045100, 0.9689885, -0.4692839, 0.9176295, -1.4221395, 1.4382725
2: -0.4292829, 1.0956014, -0.3937626, 1.0519434, -1.4812263, 1.4893640
3: -0.8731403, 1.1296451, -0.8200877, 1.0757111, -1.9488515, 1.9497328
4: -0.7632787, 1.3396218, -0.7099589, 1.2945570, -2.0578356, 2.0495806

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.2254990, 0.7289720, -0.2353991, 0.7405176, -0.9660166, 0.9643710
1: -0.4573010, 0.9075529, -0.4708995, 0.9210051, -1.3783062, 1.3784523
2: -0.3811812, 1.0387152, -0.3951180, 1.0560383, -1.4372195, 1.4338332
3: -0.8068940, 1.0590353, -0.8228747, 1.0791245, -1.8860185, 1.8819100
4: -0.6912529, 1.2759079, -0.7121502, 1.2987292, -1.9899821, 1.9880581

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1675704, 0.6571280, -1.2246854, 2.4703135, -2.6378839, 1.8818134
1: -0.3717952, 0.8276560, -1.6965811, 2.6998594, -3.0716546, 2.5242372
2: -0.2991849, 0.9327508, -1.6591611, 3.0394058, -3.3385906, 2.5919118
3: -0.6953479, 0.9345235, -2.1866965, 3.4946685, -4.1900163, 3.1212201
4: -0.5665929, 1.1308064, -2.5240049, 3.5537622, -4.1203547, 3.6548114

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500294, upper bound: 1.0488142
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500294, upper bound: 1.0488327
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.2368348, 0.7910267, -1.1696302, 2.3529849, -2.5898190, 1.9606569
1: -0.4673485, 1.0109488, -1.6229377, 2.5730762, -3.0404246, 2.6338863
2: -0.3877510, 1.1442752, -1.5897703, 2.9033613, -3.2911122, 2.7340455
3: -0.8521537, 1.1372855, -2.0951705, 3.3315568, -4.1837106, 3.2324560
4: -0.7207072, 1.3401866, -2.4250779, 3.3991196, -4.1198268, 3.7652645

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500294, upper bound: 1.0488142
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500294, upper bound: 1.0488327
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.2631326, 0.7698788, -1.2413032, 2.5691936, -2.8323259, 2.0111821
1: -0.5045100, 0.9689885, -1.7233796, 2.7961643, -3.3006742, 2.6923680
2: -0.4292829, 1.0956014, -1.6813955, 3.1583161, -3.5875990, 2.7769971
3: -0.8731403, 1.1296451, -2.2370095, 3.6173403, -4.4904804, 3.3666546
4: -0.7632787, 1.3396218, -2.5641105, 3.6751060, -4.4383845, 3.9037323

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0495637
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0496162
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.2254990, 0.7289720, -1.2413032, 2.5691936, -2.7946925, 1.9702752
1: -0.4573010, 0.9075529, -1.7233796, 2.7961643, -3.2534652, 2.6309326
2: -0.3811812, 1.0387152, -1.6813955, 3.1583161, -3.5394969, 2.7201109
3: -0.8068940, 1.0590353, -2.2370095, 3.6173403, -4.4242334, 3.2960448
4: -0.6912529, 1.2759079, -2.5641105, 3.6751060, -4.3663588, 3.8400183

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0495637
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0496162
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.2631326, 0.7698788, -1.2977409, 2.6237350, -2.8868675, 2.0676198
1: -0.5045100, 0.9689885, -1.7878509, 2.8421307, -3.3466406, 2.7568393
2: -0.4292829, 1.0956014, -1.7525644, 3.2204282, -3.6497111, 2.8481660
3: -0.8731403, 1.1296451, -2.3005936, 3.7071474, -4.5802879, 3.4302387
4: -0.7632787, 1.3396218, -2.6624808, 3.7612123, -4.5244913, 4.0021029

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0500224
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0500224
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.2254990, 0.7289720, -1.2977409, 2.6237350, -2.8492339, 2.0267129
1: -0.4573010, 0.9075529, -1.7878509, 2.8421307, -3.2994318, 2.6954038
2: -0.3811812, 1.0387152, -1.7525644, 3.2204282, -3.6016095, 2.7912798
3: -0.8068940, 1.0590353, -2.3005936, 3.7071474, -4.5140409, 3.3596289
4: -0.6912529, 1.2759079, -2.6624808, 3.7612123, -4.4524651, 3.9383888

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0498831
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0507191
time: 0.42 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.03 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.03
Output dim: 0, lower bound: -1.0502708, upper bound: 1.0502708
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.03
Output dim: 0, lower bound: -1.0502708, upper bound: 1.0502708
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.03
Output dim: 0, lower bound: -1.0502708, upper bound: 1.0502708
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.03
Output dim: 0, lower bound: -1.0502708, upper bound: 1.0502708
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.03
Output dim: 0, lower bound: -1.0502708, upper bound: 1.0508892
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.03
Output dim: 0, lower bound: -1.0502708, upper bound: 1.0502708
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.03
Output dim: 0, lower bound: -1.0502708, upper bound: 1.0508892
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.03
Output dim: 0, lower bound: -1.0505944, upper bound: 1.0508893
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.03
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0548332
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.03
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0548332
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.03
Output dim: 0, lower bound: -1.0514621, upper bound: 1.0534027
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.03
Output dim: 0, lower bound: -1.0508892, upper bound: 1.0505944
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.03
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.03
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.03
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.03
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.03
Output dim: 0, lower bound: -1.0500294, upper bound: 1.0488142
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.03
Output dim: 0, lower bound: -1.0500294, upper bound: 1.0488327
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.03
Output dim: 0, lower bound: -1.0500294, upper bound: 1.0488142
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.03
Output dim: 0, lower bound: -1.0500294, upper bound: 1.0488327
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.03
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0495637
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.03
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0496162
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.03
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0495637
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.03
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0496162
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.03
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0500224
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.03
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0500224
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.03
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0498831
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.03
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0507191

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1851257, 0.6783313, -0.1851257, 0.6783313, -0.8634571, 0.8634571
1: -0.3969364, 0.8526834, -0.3969364, 0.8526834, -1.2496197, 1.2496197
2: -0.3242711, 0.9644121, -0.3242711, 0.9644121, -1.2886832, 1.2886832
3: -0.7251614, 0.9704387, -0.7251614, 0.9704387, -1.6956002, 1.6956002
4: -0.6038694, 1.1718525, -0.6038694, 1.1718525, -1.7757219, 1.7757219

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502594, upper bound: 1.0504900
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502594, upper bound: 1.0505067
time: 0.40 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1851257, 0.6783313, -0.2674521, 0.8277271, -1.0128528, 0.9457834
1: -0.3969364, 0.8526834, -0.5075184, 1.0548140, -1.4517504, 1.3602018
2: -0.3242711, 0.9644121, -0.4327361, 1.2001319, -1.5244030, 1.3971481
3: -0.7251614, 0.9704387, -0.9017408, 1.1985904, -1.9237518, 1.8721795
4: -0.6038694, 1.1718525, -0.7864016, 1.4086894, -2.0125589, 1.9582541

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502594, upper bound: 1.0504900
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502594, upper bound: 1.0505067
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.2674521, 0.8277271, -0.1848690, 0.6776016, -0.9450537, 1.0125960
1: -0.5075184, 1.0548140, -0.3965327, 0.8516805, -1.3591989, 1.4513466
2: -0.4327361, 1.2001319, -0.3239230, 0.9632064, -1.3959424, 1.5240549
3: -0.9017408, 1.1985904, -0.7243758, 0.9695151, -1.8712559, 1.9229662
4: -0.7864016, 1.4086894, -0.6033278, 1.1706798, -1.9570814, 2.0120173

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501355, upper bound: 1.0501891
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500938, upper bound: 1.0500938
time: 0.46 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.2674521, 0.8277271, -0.2548658, 0.7943116, -1.0617638, 1.0825928
1: -0.5075184, 1.0548140, -0.4891222, 1.0074059, -1.5149243, 1.5439363
2: -0.4327361, 1.2001319, -0.4157726, 1.1434062, -1.5761423, 1.6159046
3: -0.9017408, 1.1985904, -0.8697562, 1.1547513, -2.0564921, 2.0683465
4: -0.7864016, 1.4086894, -0.7592028, 1.3576846, -2.1440864, 2.1678922

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501355, upper bound: 1.0501891
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500938, upper bound: 1.0500938
time: 0.37 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1851257, 0.6783313, -0.2122585, 0.7096910, -0.8948168, 0.8905898
1: -0.3969364, 0.8526834, -0.4381096, 0.8818882, -1.2788246, 1.2907929
2: -0.3242711, 0.9644121, -0.3631817, 1.0126708, -1.3369418, 1.3275938
3: -0.7251614, 0.9704387, -0.7779223, 1.0290604, -1.7542218, 1.7483611
4: -0.6038694, 1.1718525, -0.6656082, 1.2431200, -1.8469894, 1.8374606

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502594, upper bound: 1.0506477
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0529364, upper bound: 1.0506646
time: 0.41 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1851257, 0.6783313, -0.3022564, 0.8574660, -1.0425918, 0.9805877
1: -0.3969364, 0.8526834, -0.5588838, 1.0796481, -1.4765846, 1.4115672
2: -0.3242711, 0.9644121, -0.4826456, 1.2362378, -1.5605088, 1.4470577
3: -0.7251614, 0.9704387, -0.9591278, 1.2640674, -1.9892288, 1.9295666
4: -0.6038694, 1.1718525, -0.8620855, 1.4785453, -2.0824146, 2.0339379

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502594, upper bound: 1.0506477
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0529364, upper bound: 1.0506646
time: 0.44 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.2674521, 0.8277271, -0.2122585, 0.7096910, -0.9771431, 1.0399857
1: -0.5075184, 1.0548140, -0.4381096, 0.8818882, -1.3894066, 1.4929236
2: -0.4327361, 1.2001319, -0.3631817, 1.0126708, -1.4454069, 1.5633136
3: -0.9017408, 1.1985904, -0.7779223, 1.0290604, -1.9308012, 1.9765127
4: -0.7864016, 1.4086894, -0.6656082, 1.2431200, -2.0295215, 2.0742974

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 6

Time for candidate selection: 2.48 seconds

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0497469, upper bound: 1.0493897
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0464394, upper bound: 1.0470003
time: 0.48 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.2674521, 0.8277271, -0.3022564, 0.8574660, -1.1249182, 1.1299834
1: -0.5075184, 1.0548140, -0.5588838, 1.0796481, -1.5871665, 1.6136978
2: -0.4327361, 1.2001319, -0.4826456, 1.2362378, -1.6689739, 1.6827774
3: -0.9017408, 1.1985904, -0.9591278, 1.2640674, -2.1658082, 2.1577182
4: -0.7864016, 1.4086894, -0.8620855, 1.4785453, -2.2649469, 2.2707748

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
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
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 6

Time for candidate selection: 2.45 seconds

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0497469, upper bound: 1.0493897
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0464395, upper bound: 1.0472048
time: 0.45 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.2631326, 0.7698788, -0.2350464, 0.7415904, -1.0047231, 1.0049253
1: -0.5045100, 0.9689885, -0.4636297, 0.9372106, -1.4417207, 1.4326181
2: -0.4292829, 1.0956014, -0.3912313, 1.0568323, -1.4861152, 1.4868327
3: -0.8731403, 1.1296451, -0.8212245, 1.0728223, -1.9459627, 1.9508696
4: -0.7632787, 1.3396218, -0.7068326, 1.2774920, -2.0407708, 2.0464544

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0537152, upper bound: 1.0548257
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0528525, upper bound: 1.0538242
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 4

Time for candidate selection: 3.47 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0544690
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0545389
time: 0.40 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.2631326, 0.7698788, -0.1909873, 0.6899648, -0.9530973, 0.9608661
1: -0.5045100, 0.9689885, -0.4062611, 0.8681383, -1.3726482, 1.3752496
2: -0.4292829, 1.0956014, -0.3319388, 0.9815091, -1.4107921, 1.4275403
3: -0.8731403, 1.1296451, -0.7429183, 0.9865521, -1.8596925, 1.8725634
4: -0.7632787, 1.3396218, -0.6151925, 1.1911318, -1.9544106, 1.9548143

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0537152, upper bound: 1.0548257
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0528525, upper bound: 1.0538242
time: 0.47 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 4

Time for candidate selection: 3.58 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0544690
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0545389
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.2254990, 0.7289720, -0.1842124, 0.6772777, -0.9027767, 0.9131844
1: -0.4573010, 0.9075529, -0.3956861, 0.8514196, -1.3087206, 1.3032391
2: -0.3811812, 1.0387152, -0.3229724, 0.9628212, -1.3440025, 1.3616877
3: -0.8068940, 1.0590353, -0.7236704, 0.9685891, -1.7754831, 1.7827057
4: -0.6912529, 1.2759079, -0.6019493, 1.1698287, -1.8610816, 1.8778572

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508892, upper bound: 1.0505944
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508892, upper bound: 1.0505944
time: 0.50 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.2081584, 0.7020422, -0.2663884, 0.8262666, -1.0344250, 0.9684306
1: -0.4291762, 0.8767916, -0.5060917, 1.0530276, -1.4822038, 1.3828833
2: -0.3551885, 1.0038459, -0.4312198, 1.1978312, -1.5530196, 1.4350657
3: -0.7773868, 1.0150626, -0.8999382, 1.1961780, -1.9735647, 1.9150007
4: -0.6538678, 1.2271178, -0.7841458, 1.4060340, -2.0599017, 2.0112636

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508892, upper bound: 1.0505944
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508892, upper bound: 1.0505944
time: 0.41 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.2631326, 0.7698788, -0.2631326, 0.7698788, -1.0330114, 1.0330114
1: -0.5045100, 0.9689885, -0.5045100, 0.9689885, -1.4734986, 1.4734986
2: -0.4292829, 1.0956014, -0.4292829, 1.0956014, -1.5248843, 1.5248843
3: -0.8731403, 1.1296451, -0.8731403, 1.1296451, -2.0027854, 2.0027854
4: -0.7632787, 1.3396218, -0.7632787, 1.3396218, -2.1029005, 2.1029005

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 4

Time for candidate selection: 2.51 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0536725
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0537102
time: 0.35 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2631326, 0.7698788, -0.2254990, 0.7289720, -0.9921045, 0.9953778
1: -0.5045100, 0.9689885, -0.4573010, 0.9075529, -1.4120629, 1.4262896
2: -0.4292829, 1.0956014, -0.3811812, 1.0387152, -1.4679981, 1.4767827
3: -0.8731403, 1.1296451, -0.8068940, 1.0590353, -1.9321756, 1.9365392
4: -0.7632787, 1.3396218, -0.6912529, 1.2759079, -2.0391865, 2.0308747

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 4

Time for candidate selection: 2.60 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0536725
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0537102
time: 0.37 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.2254990, 0.7289720, -0.2631326, 0.7698788, -0.9953778, 0.9921045
1: -0.4573010, 0.9075529, -0.5045100, 0.9689885, -1.4262896, 1.4120629
2: -0.3811812, 1.0387152, -0.4292829, 1.0956014, -1.4767827, 1.4679981
3: -0.8068940, 1.0590353, -0.8731403, 1.1296451, -1.9365392, 1.9321756
4: -0.6912529, 1.2759079, -0.7632787, 1.3396218, -2.0308747, 2.0391865

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 4

Time for candidate selection: 2.48 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0536198
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0537102
time: 0.42 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.2254990, 0.7289720, -0.2254990, 0.7289720, -0.9544709, 0.9544709
1: -0.4573010, 0.9075529, -0.4573010, 0.9075529, -1.3648539, 1.3648539
2: -0.3811812, 1.0387152, -0.3811812, 1.0387152, -1.4198965, 1.4198965
3: -0.8068940, 1.0590353, -0.8068940, 1.0590353, -1.8659294, 1.8659294
4: -0.6912529, 1.2759079, -0.6912529, 1.2759079, -1.9671608, 1.9671608

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 4

Time for candidate selection: 2.59 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0536198
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0537102
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1675704, 0.6571280, -1.2108216, 2.4394033, -2.6069736, 1.8679496
1: -0.3717952, 0.8276560, -1.6775537, 2.6632435, -3.0350387, 2.5052097
2: -0.2991849, 0.9327508, -1.6431754, 3.0000558, -3.2992406, 2.5759263
3: -0.6953479, 0.9345235, -2.1599312, 3.4501929, -4.1455407, 3.0944548
4: -0.5665929, 1.1308064, -2.5004611, 3.5102849, -4.0768776, 3.6312675

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0529566, upper bound: 1.0500381
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0529566, upper bound: 1.0500381
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1675704, 0.6571280, -1.1420137, 2.3529160, -2.5204864, 1.7991416
1: -0.3717952, 0.8276560, -1.5802696, 2.6097629, -2.9815578, 2.4079256
2: -0.2991849, 0.9327508, -1.5261636, 2.9508667, -3.2500510, 2.4589143
3: -0.6953479, 0.9345235, -2.0956416, 3.3117151, -4.0070629, 3.0301652
4: -0.5665929, 1.1308064, -2.3511953, 3.4049644, -3.9715569, 3.4820018

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0529566, upper bound: 1.0500381
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0529566, upper bound: 1.0500381
time: 0.34 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.2368348, 0.7910267, -1.2108216, 2.4394033, -2.6762381, 2.0018482
1: -0.4673485, 1.0109488, -1.6775537, 2.6632435, -3.1305916, 2.6885023
2: -0.3877510, 1.1442752, -1.6431754, 3.0000558, -3.3878069, 2.7874506
3: -0.8521537, 1.1372855, -2.1599312, 3.4501929, -4.3023467, 3.2972167
4: -0.7207072, 1.3401866, -2.5004611, 3.5102849, -4.2309923, 3.8406477

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500294, upper bound: 1.0488142
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500294, upper bound: 1.0488142
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.2368348, 0.7910267, -1.1420137, 2.3529160, -2.5897505, 1.9330404
1: -0.4673485, 1.0109488, -1.5802696, 2.6097629, -3.0771110, 2.5912185
2: -0.3877510, 1.1442752, -1.5261636, 2.9508667, -3.3386176, 2.6704388
3: -0.8521537, 1.1372855, -2.0956416, 3.3117151, -4.1638689, 3.2329271
4: -0.7207072, 1.3401866, -2.3511953, 3.4049644, -4.1256714, 3.6913819

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500294, upper bound: 1.0488142
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500294, upper bound: 1.0488142
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.2631326, 0.7698788, -1.2317941, 2.5485647, -2.8116972, 2.0016730
1: -0.5045100, 0.9689885, -1.7097392, 2.7696524, -3.2741623, 2.6787276
2: -0.4292829, 1.0956014, -1.6688566, 3.1283512, -3.5576341, 2.7644582
3: -0.8731403, 1.1296451, -2.2169631, 3.5870750, -4.4602156, 3.3466082
4: -0.7632787, 1.3396218, -2.5448000, 3.6422343, -4.4055128, 3.8844218

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 4

Time for candidate selection: 2.45 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0491880
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0492286
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.2631326, 0.7698788, -1.3011413, 2.5799091, -2.8430414, 2.0710201
1: -0.5045100, 0.9689885, -1.7600503, 2.8037138, -3.3082237, 2.7290387
2: -0.4292829, 1.0956014, -1.7155523, 3.1846526, -3.6139355, 2.8111539
3: -0.8731403, 1.1296451, -2.2817159, 3.6491742, -4.5223141, 3.4113610
4: -0.7632787, 1.3396218, -2.6121106, 3.7078054, -4.4710827, 3.9517324

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 4

Time for candidate selection: 2.47 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0492215
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0492755
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.2254990, 0.7289720, -1.2317941, 2.5485647, -2.7740636, 1.9607661
1: -0.4573010, 0.9075529, -1.7097392, 2.7696524, -3.2269535, 2.6172922
2: -0.3811812, 1.0387152, -1.6688566, 3.1283512, -3.5095320, 2.7075720
3: -0.8068940, 1.0590353, -2.2169631, 3.5870750, -4.3939691, 3.2759984
4: -0.6912529, 1.2759079, -2.5448000, 3.6422343, -4.3334870, 3.8207078

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 4

Time for candidate selection: 2.47 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0491383
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0492286
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.2254990, 0.7289720, -1.3011413, 2.5799091, -2.8054080, 2.0301132
1: -0.4573010, 0.9075529, -1.7600503, 2.8037138, -3.2610145, 2.6676033
2: -0.3811812, 1.0387152, -1.7155523, 3.1846526, -3.5658336, 2.7542677
3: -0.8068940, 1.0590353, -2.2817159, 3.6491742, -4.4560671, 3.3407512
4: -0.6912529, 1.2759079, -2.6121106, 3.7078054, -4.3990583, 3.8880186

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 4

Time for candidate selection: 2.66 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0491851
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0492755
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.2631326, 0.7698788, -1.2890360, 2.6080267, -2.8711593, 2.0589149
1: -0.5045100, 0.9689885, -1.7757368, 2.8224382, -3.3269482, 2.7447252
2: -0.4292829, 1.0956014, -1.7412896, 3.1970921, -3.6263750, 2.8368912
3: -0.8731403, 1.1296451, -2.2828507, 3.6829684, -4.5561085, 3.4124959
4: -0.7632787, 1.3396218, -2.6450953, 3.7347975, -4.4980764, 3.9847171

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0510389, upper bound: 1.0491357
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 4

Time for candidate selection: 3.02 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0496647
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0496745
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2631326, 0.7698788, -1.3504868, 2.6251640, -2.8882966, 2.1203656
1: -0.5045100, 0.9689885, -1.8179705, 2.8394029, -3.3439131, 2.7869589
2: -0.4292829, 1.0956014, -1.7806034, 3.2355962, -3.6648791, 2.8762050
3: -0.8731403, 1.1296451, -2.3383503, 3.7266910, -4.5998311, 3.4679954
4: -0.7632787, 1.3396218, -2.7018437, 3.7814958, -4.5447741, 4.0414658

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0510389, upper bound: 1.0492629
time: 0.48 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 4

Time for candidate selection: 3.21 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0499589
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0499313
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.2254990, 0.7289720, -1.2890360, 2.6080267, -2.8335257, 2.0180080
1: -0.4573010, 0.9075529, -1.7757368, 2.8224382, -3.2797394, 2.6832898
2: -0.3811812, 1.0387152, -1.7412896, 3.1970921, -3.5782728, 2.7800050
3: -0.8068940, 1.0590353, -2.2828507, 3.6829684, -4.4898624, 3.3418860
4: -0.6912529, 1.2759079, -2.6450953, 3.7347975, -4.4260502, 3.9210033

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0469137, upper bound: 1.0479783
time: 0.44 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0486414, upper bound: 1.0483880
time: 0.44 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.2254990, 0.7289720, -1.3504868, 2.6251640, -2.8506629, 2.0794587
1: -0.4573010, 0.9075529, -1.8179705, 2.8394029, -3.2967038, 2.7255235
2: -0.3811812, 1.0387152, -1.7806034, 3.2355962, -3.6167774, 2.8193188
3: -0.8068940, 1.0590353, -2.3383503, 3.7266910, -4.5335836, 3.3973856
4: -0.6912529, 1.2759079, -2.7018437, 3.7814958, -4.4727488, 3.9777517

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0469137, upper bound: 1.0506330
time: 0.48 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0486414, upper bound: 1.0506391
time: 0.41 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 2.71 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0502594, upper bound: 1.0504900
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0502594, upper bound: 1.0505067
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0502594, upper bound: 1.0504900
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0502594, upper bound: 1.0505067
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0501355, upper bound: 1.0501891
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0500938, upper bound: 1.0500938
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0501355, upper bound: 1.0501891
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0500938, upper bound: 1.0500938
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0502594, upper bound: 1.0506477
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0529364, upper bound: 1.0506646
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0502594, upper bound: 1.0506477
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0529364, upper bound: 1.0506646
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0497469, upper bound: 1.0493897
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0464394, upper bound: 1.0470003
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0497469, upper bound: 1.0493897
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0464395, upper bound: 1.0472048
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0544690
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0545389
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0544690
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0545389
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0508892, upper bound: 1.0505944
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0508892, upper bound: 1.0505944
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0508892, upper bound: 1.0505944
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0508892, upper bound: 1.0505944
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0536725
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0537102
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0536725
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0537102
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0536198
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0537102
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0536198
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0537102
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0529566, upper bound: 1.0500381
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0529566, upper bound: 1.0500381
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0529566, upper bound: 1.0500381
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0529566, upper bound: 1.0500381
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0500294, upper bound: 1.0488142
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0500294, upper bound: 1.0488142
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0500294, upper bound: 1.0488142
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0500294, upper bound: 1.0488142
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0491880
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0492286
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0492215
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0492755
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0491383
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0492286
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0491851
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0492755
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0496647
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0496745
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0499589
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0499313
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0469137, upper bound: 1.0479783
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0486414, upper bound: 1.0483880
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0469137, upper bound: 1.0506330
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.71
Output dim: 0, lower bound: -1.0486414, upper bound: 1.0506391

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1546751, 0.6344486, -0.1774142, 0.6674564, -0.8221315, 0.8118628
1: -0.3508692, 0.8027984, -0.3853742, 0.8408388, -1.1917080, 1.1881726
2: -0.2792361, 0.8971295, -0.3130327, 0.9479907, -1.2272267, 1.2101622
3: -0.6675518, 0.9013090, -0.7106574, 0.9531208, -1.6206726, 1.6119664
4: -0.5352525, 1.0835073, -0.5868486, 1.1500280, -1.6852804, 1.6703558

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0529589, upper bound: 1.0544141
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0529589, upper bound: 1.0543839
time: 0.42 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1633435, 0.6488099, -0.1837167, 0.6764191, -0.8397626, 0.8325267
1: -0.3648036, 0.8238323, -0.3948803, 0.8508478, -1.2156514, 1.2187126
2: -0.2912067, 0.9218535, -0.3221599, 0.9616230, -1.2528298, 1.2440133
3: -0.6862502, 0.9238169, -0.7225855, 0.9674332, -1.6536834, 1.6464024
4: -0.5532730, 1.1132669, -0.6006362, 1.1680149, -1.7212878, 1.7139032

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0547677, upper bound: 1.0547700
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0547694, upper bound: 1.0547694
time: 0.38 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1546751, 0.6344486, -0.2593745, 0.8156036, -0.9702787, 0.8938231
1: -0.3508692, 0.8027984, -0.4961942, 1.0417651, -1.3926343, 1.2989926
2: -0.2792361, 0.8971295, -0.4210599, 1.1825309, -1.4617670, 1.3181894
3: -0.6675518, 0.9013090, -0.8866759, 1.1798939, -1.8474456, 1.7879848
4: -0.5352525, 1.0835073, -0.7686534, 1.3852768, -1.9205292, 1.8521607

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 6

Time for candidate selection: 2.52 seconds

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0481153, upper bound: 1.0485276
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0454206, upper bound: 1.0453928
time: 0.41 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1633435, 0.6488099, -0.2659268, 0.8255982, -0.9889417, 0.9147367
1: -0.3648036, 0.8238323, -0.5052903, 1.0527209, -1.4175245, 1.3291225
2: -0.2912067, 0.9218535, -0.4304219, 1.1970170, -1.4882237, 1.3522754
3: -0.6862502, 0.9238169, -0.8989245, 1.1952195, -1.8814697, 1.8227414
4: -0.5532730, 1.1132669, -0.7828714, 1.4042351, -1.9575081, 1.8961383

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0529364, upper bound: 1.0503436
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0529278, upper bound: 1.0503142
time: 0.45 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1829112, 0.7200209, -0.1675704, 0.6571280, -0.8400392, 0.8875914
1: -0.3864989, 0.9190384, -0.3717952, 0.8276560, -1.2141550, 1.2908336
2: -0.3115277, 1.0421524, -0.2991849, 0.9327508, -1.2442786, 1.3413372
3: -0.7443157, 1.0138530, -0.6953479, 0.9345235, -1.6788392, 1.7092009
4: -0.6034802, 1.2099588, -0.5665929, 1.1308064, -1.7342867, 1.7765517

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505969, upper bound: 1.0523658
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505969, upper bound: 1.0532118
time: 0.41 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.2494289, 0.7944719, -0.1777069, 0.6674852, -0.9169142, 0.9721788
1: -0.4813044, 1.0119214, -0.3858544, 0.8395716, -1.3208760, 1.3977758
2: -0.4077803, 1.1496273, -0.3133788, 0.9491947, -1.3569750, 1.4630061
3: -0.8642187, 1.1482675, -0.7111726, 0.9526974, -1.8169160, 1.8594401
4: -0.7469358, 1.3539220, -0.5874691, 1.1527991, -1.8997350, 1.9413911

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505969, upper bound: 1.0523658
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505969, upper bound: 1.0532118
time: 0.40 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1829112, 0.7200209, -0.2303006, 0.7634068, -0.9463180, 0.9503215
1: -0.3864989, 0.9190384, -0.4565862, 0.9706300, -1.3571290, 1.3756247
2: -0.3115277, 1.0421524, -0.3796775, 1.0976576, -1.4091853, 1.4218299
3: -0.7443157, 1.0138530, -0.8286581, 1.1033072, -1.8476230, 1.8425111
4: -0.6034802, 1.2099588, -0.7059905, 1.3008198, -1.9043000, 1.9159493

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500938, upper bound: 1.0500938
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500938, upper bound: 1.0500938
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.2494289, 0.7944719, -0.2499463, 0.7862649, -1.0356939, 1.0444181
1: -0.4813044, 1.0119214, -0.4822609, 0.9972249, -1.4785292, 1.4941823
2: -0.4077803, 1.1496273, -0.4089095, 1.1315844, -1.5393647, 1.5585368
3: -0.8642187, 1.1482675, -0.8603870, 1.1421520, -2.0063705, 2.0086546
4: -0.7469358, 1.3539220, -0.7485338, 1.3442633, -2.0911992, 2.1024559

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500938, upper bound: 1.0500938
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500938, upper bound: 1.0500938
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1546751, 0.6344486, -0.2041048, 0.6977630, -0.8524380, 0.8385535
1: -0.3508692, 0.8027984, -0.4260215, 0.8692849, -1.2201540, 1.2288198
2: -0.2792361, 0.8971295, -0.3512143, 0.9951268, -1.2743628, 1.2483438
3: -0.6675518, 0.9013090, -0.7626776, 1.0101199, -1.6776717, 1.6639864
4: -0.5352525, 1.0835073, -0.6474289, 1.2196317, -1.7548841, 1.7309363

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0528513, upper bound: 1.0520721
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0528721, upper bound: 1.0520802
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0528721, upper bound: 1.0520802
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1633435, 0.6488099, -0.2108487, 0.7076982, -0.8710417, 0.8596587
1: -0.3648036, 0.8238323, -0.4360059, 0.8799497, -1.2447534, 1.2598382
2: -0.2912067, 0.9218535, -0.3610466, 1.0098436, -1.3010503, 1.2829001
3: -0.6862502, 0.9238169, -0.7752968, 1.0258479, -1.7120981, 1.6991136
4: -0.5532730, 1.1132669, -0.6623644, 1.2392564, -1.7925293, 1.7756313

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0528513, upper bound: 1.0520685
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0542550, upper bound: 1.0520802
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0542550, upper bound: 1.0520802
time: 0.36 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1546751, 0.6344486, -0.2941689, 0.8452820, -0.9999570, 0.9286175
1: -0.3508692, 0.8027984, -0.5476846, 1.0665569, -1.4174261, 1.3504829
2: -0.2792361, 0.8971295, -0.4709241, 1.2176194, -1.4968555, 1.3680537
3: -0.6675518, 0.9013090, -0.9439932, 1.2445780, -1.9121298, 1.8453021
4: -0.5352525, 1.0835073, -0.8442773, 1.4537967, -1.9890492, 1.9277847

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0511374, upper bound: 1.0504673
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0511374, upper bound: 1.0506477
time: 0.41 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1633435, 0.6488099, -0.3007836, 0.8553913, -1.0187348, 0.9495935
1: -0.3648036, 0.8238323, -0.5566587, 1.0776916, -1.4424951, 1.3804910
2: -0.2912067, 0.9218535, -0.4803994, 1.2331573, -1.5243640, 1.4022529
3: -0.6862502, 0.9238169, -0.9563999, 1.2606578, -1.9469080, 1.8802167
4: -0.5532730, 1.1132669, -0.8586895, 1.4743117, -2.0275846, 1.9719565

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0529364, upper bound: 1.0505082
time: 0.51 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0526369, upper bound: 1.0504832
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0526369, upper bound: 1.0506646
time: 0.45 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.2343636, 0.7751515, -0.2122585, 0.7096910, -0.9440545, 0.9874101
1: -0.4597959, 0.9859009, -0.4381096, 0.8818882, -1.3416841, 1.4240105
2: -0.3856044, 1.1241981, -0.3631817, 1.0126708, -1.3982751, 1.4873798
3: -0.8403842, 1.1136440, -0.7779223, 1.0290604, -1.8694446, 1.8915663
4: -0.7135836, 1.3228402, -0.6656082, 1.2431200, -1.9567037, 1.9884484

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0495026, upper bound: 1.0488463
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0494713, upper bound: 1.0485787
time: 0.49 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0494312, upper bound: 1.0483198
time: 0.43 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.2343636, 0.7751515, -0.3022564, 0.8574660, -1.0918295, 1.0774078
1: -0.4597959, 0.9859009, -0.5588838, 1.0796481, -1.5394440, 1.5447848
2: -0.3856044, 1.1241981, -0.4826456, 1.2362378, -1.6218421, 1.6068437
3: -0.8403842, 1.1136440, -0.9591278, 1.2640674, -2.1044517, 2.0727718
4: -0.7135836, 1.3228402, -0.8620855, 1.4785453, -2.1921289, 2.1849256

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0495026, upper bound: 1.0486466
time: 0.45 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0494710, upper bound: 1.0485787
time: 0.46 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0494312, upper bound: 1.0482690
time: 0.43 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1981437, 0.6735660, -0.2350464, 0.7415904, -0.9397342, 0.9086125
1: -0.4075772, 0.8701689, -0.4636297, 0.9372106, -1.3447878, 1.3337986
2: -0.3290880, 0.9651487, -0.3912313, 1.0568323, -1.3859203, 1.3563800
3: -0.7565911, 0.9793513, -0.8212245, 1.0728223, -1.8294134, 1.8005757
4: -0.6072877, 1.1647243, -0.7068326, 1.2774920, -1.8847797, 1.8715570

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 4

Time for candidate selection: 2.72 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0544018
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0544018
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.2620909, 0.7696464, -0.2350464, 0.7415904, -1.0036813, 1.0046928
1: -0.5045343, 0.9919055, -0.4636297, 0.9372106, -1.4417449, 1.4555352
2: -0.4200620, 1.0960678, -0.3912313, 1.0568323, -1.4768944, 1.4872991
3: -0.8876588, 1.1395187, -0.8212245, 1.0728223, -1.9604812, 1.9607432
4: -0.7470124, 1.3261604, -0.7068326, 1.2774920, -2.0245044, 2.0329931

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 4

Time for candidate selection: 2.83 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0545392
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0545392
time: 0.43 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1981437, 0.6735660, -0.1909873, 0.6899648, -0.8881085, 0.8645533
1: -0.4075772, 0.8701689, -0.4062611, 0.8681383, -1.2757154, 1.2764300
2: -0.3290880, 0.9651487, -0.3319388, 0.9815091, -1.3105972, 1.2970874
3: -0.7565911, 0.9793513, -0.7429183, 0.9865521, -1.7431432, 1.7222695
4: -0.6072877, 1.1647243, -0.6151925, 1.1911318, -1.7984195, 1.7799169

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 4

Time for candidate selection: 2.61 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0534783, upper bound: 1.0543958
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0534783, upper bound: 1.0543985
time: 0.37 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.2620909, 0.7696464, -0.1909873, 0.6899648, -0.9520556, 0.9606337
1: -0.5045343, 0.9919055, -0.4062611, 0.8681383, -1.3726726, 1.3981667
2: -0.4200620, 1.0960678, -0.3319388, 0.9815091, -1.4015712, 1.4280066
3: -0.8876588, 1.1395187, -0.7429183, 0.9865521, -1.8742108, 1.8824370
4: -0.7470124, 1.3261604, -0.6151925, 1.1911318, -1.9381442, 1.9413530

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 4

Time for candidate selection: 2.58 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0534783, upper bound: 1.0545389
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0534783, upper bound: 1.0545389
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.2013567, 0.6970940, -0.1842124, 0.6772777, -0.8786345, 0.8813065
1: -0.4231896, 0.8673437, -0.3956861, 0.8514196, -1.2746092, 1.2630298
2: -0.3478338, 0.9940906, -0.3229724, 0.9628212, -1.3106549, 1.3170630
3: -0.7603652, 1.0071222, -0.7236704, 0.9685891, -1.7289543, 1.7307925
4: -0.6426137, 1.2188928, -0.6019493, 1.1698287, -1.8124423, 1.8208420

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506477, upper bound: 1.0511374
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506646, upper bound: 1.0529364
time: 0.47 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.2909008, 0.8438774, -0.1842124, 0.6772777, -0.9681785, 1.0280899
1: -0.5446191, 1.0634959, -0.3956861, 0.8514196, -1.3960388, 1.4591820
2: -0.4665691, 1.2150321, -0.3229724, 0.9628212, -1.4293903, 1.5380045
3: -0.9410769, 1.2407138, -0.7236704, 0.9685891, -1.9096661, 1.9643842
4: -0.8382176, 1.4520541, -0.6019493, 1.1698287, -2.0080462, 2.0540035

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506477, upper bound: 1.0511374
time: 0.45 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506646, upper bound: 1.0529364
time: 0.43 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.2013567, 0.6970940, -0.2663884, 0.8262666, -1.0276233, 0.9634825
1: -0.4231896, 0.8673437, -0.5060917, 1.0530276, -1.4762173, 1.3734354
2: -0.3478338, 0.9940906, -0.4312198, 1.1978312, -1.5456649, 1.4253104
3: -0.7603652, 1.0071222, -0.8999382, 1.1961780, -1.9565432, 1.9070604
4: -0.6426137, 1.2188928, -0.7841458, 1.4060340, -2.0486476, 2.0030386

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 6

Time for candidate selection: 2.58 seconds

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0493897, upper bound: 1.0497469
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0467689, upper bound: 1.0464394
time: 0.49 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.2909008, 0.8438774, -0.2663884, 0.8262666, -1.1171674, 1.1102659
1: -0.5446191, 1.0634959, -0.5060917, 1.0530276, -1.5976467, 1.5695876
2: -0.4665691, 1.2150321, -0.4312198, 1.1978312, -1.6644003, 1.6462519
3: -0.9410769, 1.2407138, -0.8999382, 1.1961780, -2.1372550, 2.1406519
4: -0.8382176, 1.4520541, -0.7841458, 1.4060340, -2.2442517, 2.2361999

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 6

Time for candidate selection: 2.62 seconds

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0493897, upper bound: 1.0497469
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0467689, upper bound: 1.0465156
time: 0.42 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1981437, 0.6735660, -0.2631326, 0.7698788, -0.9680225, 0.9366986
1: -0.4075772, 0.8701689, -0.5045100, 0.9689885, -1.3765657, 1.3746790
2: -0.3290880, 0.9651487, -0.4292829, 1.0956014, -1.4246894, 1.3944316
3: -0.7565911, 0.9793513, -0.8731403, 1.1296451, -1.8862362, 1.8524916
4: -0.6072877, 1.1647243, -0.7632787, 1.3396218, -1.9469094, 1.9280031

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 4

Time for candidate selection: 2.61 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0535787
time: 0.46 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0535787
time: 0.44 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.2620909, 0.7696464, -0.2631326, 0.7698788, -1.0319697, 1.0327790
1: -0.5045343, 0.9919055, -0.5045100, 0.9689885, -1.4735228, 1.4964156
2: -0.4200620, 1.0960678, -0.4292829, 1.0956014, -1.5156634, 1.5253507
3: -0.8876588, 1.1395187, -0.8731403, 1.1296451, -2.0173039, 2.0126591
4: -0.7470124, 1.3261604, -0.7632787, 1.3396218, -2.0866342, 2.0894392

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 4

Time for candidate selection: 2.63 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0537102
time: 0.47 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0537102
time: 0.40 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1981437, 0.6735660, -0.2254990, 0.7289720, -0.9271157, 0.8990650
1: -0.4075772, 0.8701689, -0.4573010, 0.9075529, -1.3151300, 1.3274699
2: -0.3290880, 0.9651487, -0.3811812, 1.0387152, -1.3678032, 1.3463299
3: -0.7565911, 0.9793513, -0.8068940, 1.0590353, -1.8156264, 1.7862453
4: -0.6072877, 1.1647243, -0.6912529, 1.2759079, -1.8831956, 1.8559773

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 4

Time for candidate selection: 2.70 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0534783, upper bound: 1.0535787
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0534783, upper bound: 1.0535787
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.2620909, 0.7696464, -0.2254990, 0.7289720, -0.9910628, 0.9951454
1: -0.5045343, 0.9919055, -0.4573010, 0.9075529, -1.4120872, 1.4492066
2: -0.4200620, 1.0960678, -0.3811812, 1.0387152, -1.4587772, 1.4772490
3: -0.8876588, 1.1395187, -0.8068940, 1.0590353, -1.9466941, 1.9464128
4: -0.7470124, 1.3261604, -0.6912529, 1.2759079, -2.0229201, 2.0174134

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 4

Time for candidate selection: 2.72 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0534783, upper bound: 1.0537102
time: 0.45 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0534783, upper bound: 1.0537102
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1720736, 0.6441831, -0.2631326, 0.7698788, -0.9419523, 0.9073157
1: -0.3784128, 0.8175137, -0.5045100, 0.9689885, -1.3474014, 1.3220237
2: -0.3015163, 0.9219700, -0.4292829, 1.0956014, -1.3971177, 1.3512529
3: -0.7047867, 0.9291709, -0.8731403, 1.1296451, -1.8344318, 1.8023112
4: -0.5681181, 1.1211691, -0.7632787, 1.3396218, -1.9077399, 1.8844478

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0514589, upper bound: 1.0516042
time: 0.52 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 4

Time for candidate selection: 3.16 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0535655
time: 0.45 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0535655
time: 0.44 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.2310074, 0.7175490, -0.2631326, 0.7698788, -1.0008862, 0.9806816
1: -0.4606035, 0.9202803, -0.5045100, 0.9689885, -1.4295919, 1.4247904
2: -0.3788169, 1.0198433, -0.4292829, 1.0956014, -1.4744184, 1.4491262
3: -0.8183979, 1.0579665, -0.8731403, 1.1296451, -1.9480430, 1.9311068
4: -0.6781206, 1.2412384, -0.7632787, 1.3396218, -2.0177424, 2.0045171

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0518665, upper bound: 1.0524409
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 4

Time for candidate selection: 3.15 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0537102
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0537102
time: 0.44 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1720736, 0.6441831, -0.2254990, 0.7289720, -0.9010456, 0.8696821
1: -0.3784128, 0.8175137, -0.4573010, 0.9075529, -1.2859657, 1.2748147
2: -0.3015163, 0.9219700, -0.3811812, 1.0387152, -1.3402315, 1.3031512
3: -0.7047867, 0.9291709, -0.8068940, 1.0590353, -1.7638220, 1.7360649
4: -0.5681181, 1.1211691, -0.6912529, 1.2759079, -1.8440260, 1.8124220

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0514589, upper bound: 1.0516432
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512863, upper bound: 1.0509469
time: 0.44 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.2310074, 0.7175490, -0.2254990, 0.7289720, -0.9599794, 0.9430480
1: -0.4606035, 0.9202803, -0.4573010, 0.9075529, -1.3681564, 1.3775814
2: -0.3788169, 1.0198433, -0.3811812, 1.0387152, -1.4175322, 1.4010246
3: -0.8183979, 1.0579665, -0.8068940, 1.0590353, -1.8774332, 1.8648605
4: -0.6781206, 1.2412384, -0.6912529, 1.2759079, -1.9540285, 1.9324913

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0515344, upper bound: 1.0529024
time: 0.51 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512377, upper bound: 1.0513255
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1476599, 0.6317843, -1.2108216, 2.4394033, -2.5870631, 1.8426059
1: -0.3481481, 0.7843444, -1.6775537, 2.6632435, -3.0113916, 2.4618981
2: -0.2732941, 0.8929424, -1.6431754, 3.0000558, -3.2733498, 2.5361178
3: -0.6720750, 0.8882146, -2.1599312, 3.4501929, -4.1222677, 3.0481458
4: -0.5259104, 1.0916746, -2.5004611, 3.5102849, -4.0361953, 3.5921357

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0517931, upper bound: 1.0490584
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0533510, upper bound: 1.0502633
time: 0.45 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1612040, 0.6436090, -1.2108216, 2.4394033, -2.6006074, 1.8544307
1: -0.3608073, 0.8110970, -1.6775537, 2.6632435, -3.0240507, 2.4886506
2: -0.2890804, 0.9154254, -1.6431754, 3.0000558, -3.2891359, 2.5586007
3: -0.6798525, 0.9134842, -2.1599312, 3.4501929, -4.1300454, 3.0734153
4: -0.5504088, 1.1098173, -2.5004611, 3.5102849, -4.0606937, 3.6102784

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0517931, upper bound: 1.0490584
time: 0.47 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0533510, upper bound: 1.0502633
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1476599, 0.6317843, -1.1420137, 2.3529160, -2.5005758, 1.7737980
1: -0.3481481, 0.7843444, -1.5802696, 2.6097629, -2.9579110, 2.3646140
2: -0.2732941, 0.8929424, -1.5261636, 2.9508667, -3.2241602, 2.4191060
3: -0.6720750, 0.8882146, -2.0956416, 3.3117151, -3.9837902, 2.9838562
4: -0.5259104, 1.0916746, -2.3511953, 3.4049644, -3.9308746, 3.4428699

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0518830, upper bound: 1.0494476
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0514500, upper bound: 1.0487686
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1612040, 0.6436090, -1.1420137, 2.3529160, -2.5141201, 1.7856227
1: -0.3608073, 0.8110970, -1.5802696, 2.6097629, -2.9705701, 2.3913665
2: -0.2890804, 0.9154254, -1.5261636, 2.9508667, -3.2399464, 2.4415889
3: -0.6798525, 0.9134842, -2.0956416, 3.3117151, -3.9915676, 3.0091257
4: -0.5504088, 1.1098173, -2.3511953, 3.4049644, -3.9553728, 3.4610126

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0518830, upper bound: 1.0494476
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0514500, upper bound: 1.0487686
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1829112, 0.7200095, -1.2108216, 2.4394033, -2.6223145, 1.9308312
1: -0.3864852, 0.9190384, -1.6775537, 2.6632435, -3.0497286, 2.5965919
2: -0.3115277, 1.0421143, -1.6431754, 3.0000558, -3.3115835, 2.6852896
3: -0.7443157, 1.0138434, -2.1599312, 3.4501929, -4.1945086, 3.1737747
4: -0.6034802, 1.2099022, -2.5004611, 3.5102849, -4.1137648, 3.7103634

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0494356, upper bound: 1.0485050
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0494356, upper bound: 1.0488221
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.2362912, 0.7876614, -1.2108216, 2.4394033, -2.6756945, 1.9984831
1: -0.4658278, 1.0048010, -1.6775537, 2.6632435, -3.1290710, 2.6823547
2: -0.3878378, 1.1400790, -1.6431754, 3.0000558, -3.3878937, 2.7832544
3: -0.8500229, 1.1328778, -2.1599312, 3.4501929, -4.3002157, 3.2928090
4: -0.7198430, 1.3383868, -2.5004611, 3.5102849, -4.2301278, 3.8388479

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0494356, upper bound: 1.0485050
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0494356, upper bound: 1.0488221
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1829112, 0.7200095, -1.1420137, 2.3529160, -2.5358269, 1.8620231
1: -0.3864852, 0.9190384, -1.5802696, 2.6097629, -2.9962475, 2.4993081
2: -0.3115277, 1.0421143, -1.5261636, 2.9508667, -3.2623944, 2.5682778
3: -0.7443157, 1.0138434, -2.0956416, 3.3117151, -4.0560308, 3.1094851
4: -0.6034802, 1.2099022, -2.3511953, 3.4049644, -4.0084443, 3.5610976

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0496940, upper bound: 1.0470154
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0497125, upper bound: 1.0480907
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.2362912, 0.7876614, -1.1420137, 2.3529160, -2.5892072, 1.9296751
1: -0.4658278, 1.0048010, -1.5802696, 2.6097629, -3.0755899, 2.5850706
2: -0.3878378, 1.1400790, -1.5261636, 2.9508667, -3.3387046, 2.6662426
3: -0.8500229, 1.1328778, -2.0956416, 3.3117151, -4.1617379, 3.2285194
4: -0.7198430, 1.3383868, -2.3511953, 3.4049644, -4.1248069, 3.6895821

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0496940, upper bound: 1.0470154
time: 0.44 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0497125, upper bound: 1.0480907
time: 0.49 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1981437, 0.6735660, -1.2317941, 2.5485647, -2.7467084, 1.9053602
1: -0.4075772, 0.8701689, -1.7097392, 2.7696524, -3.1772294, 2.5799081
2: -0.3290880, 0.9651487, -1.6688566, 3.1283512, -3.4574392, 2.6340053
3: -0.7565911, 0.9793513, -2.2169631, 3.5870750, -4.3436656, 3.1963143
4: -0.6072877, 1.1647243, -2.5448000, 3.6422343, -4.2495217, 3.7095244

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 4

Time for candidate selection: 2.76 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507335, upper bound: 1.0491139
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507335, upper bound: 1.0491139
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.2620909, 0.7696464, -1.2317941, 2.5485647, -2.8106556, 2.0014405
1: -0.5045343, 0.9919055, -1.7097392, 2.7696524, -3.2741861, 2.7016447
2: -0.4200620, 1.0960678, -1.6688566, 3.1283512, -3.5484128, 2.7649245
3: -0.8876588, 1.1395187, -2.2169631, 3.5870750, -4.4747338, 3.3564818
4: -0.7470124, 1.3261604, -2.5448000, 3.6422343, -4.3892469, 3.8709605

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 4

Time for candidate selection: 2.79 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507335, upper bound: 1.0492494
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507335, upper bound: 1.0492494
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1981437, 0.6735660, -1.3011413, 2.5799091, -2.7780528, 1.9747074
1: -0.4075772, 0.8701689, -1.7600503, 2.8037138, -3.2112908, 2.6302192
2: -0.3290880, 0.9651487, -1.7155523, 3.1846526, -3.5137405, 2.6807010
3: -0.7565911, 0.9793513, -2.2817159, 3.6491742, -4.4057636, 3.2610672
4: -0.6072877, 1.1647243, -2.6121106, 3.7078054, -4.3150930, 3.7768350

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 4

Time for candidate selection: 2.83 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508041, upper bound: 1.0491440
time: 0.47 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508041, upper bound: 1.0491440
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.2620909, 0.7696464, -1.3011413, 2.5799091, -2.8420000, 2.0707877
1: -0.5045343, 0.9919055, -1.7600503, 2.8037138, -3.3082476, 2.7519557
2: -0.4200620, 1.0960678, -1.7155523, 3.1846526, -3.6047144, 2.8116202
3: -0.8876588, 1.1395187, -2.2817159, 3.6491742, -4.5368319, 3.4212346
4: -0.7470124, 1.3261604, -2.6121106, 3.7078054, -4.4548173, 3.9382710

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 4

Time for candidate selection: 2.69 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507335, upper bound: 1.0492755
time: 0.44 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507335, upper bound: 1.0492755
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1720736, 0.6441831, -1.2317941, 2.5485647, -2.7206383, 1.8759773
1: -0.3784128, 0.8175137, -1.7097392, 2.7696524, -3.1480651, 2.5272529
2: -0.3015163, 0.9219700, -1.6688566, 3.1283512, -3.4298673, 2.5908265
3: -0.7047867, 0.9291709, -2.2169631, 3.5870750, -4.2918615, 3.1461339
4: -0.5681181, 1.1211691, -2.5448000, 3.6422343, -4.2103524, 3.6659691

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 4

Time for candidate selection: 2.78 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507335, upper bound: 1.0491047
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507335, upper bound: 1.0491047
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.2310074, 0.7175490, -1.2317941, 2.5485647, -2.7795720, 1.9493432
1: -0.4606035, 0.9202803, -1.7097392, 2.7696524, -3.2302556, 2.6300194
2: -0.3788169, 1.0198433, -1.6688566, 3.1283512, -3.5071678, 2.6887000
3: -0.8183979, 1.0579665, -2.2169631, 3.5870750, -4.4054728, 3.2749295
4: -0.6781206, 1.2412384, -2.5448000, 3.6422343, -4.3203549, 3.7860384

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0472941, upper bound: 1.0478360
time: 0.48 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0439802, upper bound: 1.0464774
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1720736, 0.6441831, -1.3011413, 2.5799091, -2.7519822, 1.9453244
1: -0.3784128, 0.8175137, -1.7600503, 2.8037138, -3.1821265, 2.5775640
2: -0.3015163, 0.9219700, -1.7155523, 3.1846526, -3.4861686, 2.6375222
3: -0.7047867, 0.9291709, -2.2817159, 3.6491742, -4.3539600, 3.2108867
4: -0.5681181, 1.1211691, -2.6121106, 3.7078054, -4.2759228, 3.7332797

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 4

Time for candidate selection: 2.92 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507335, upper bound: 1.0491308
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507335, upper bound: 1.0491308
time: 0.48 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.2310074, 0.7175490, -1.3011413, 2.5799091, -2.8109164, 2.0186903
1: -0.4606035, 0.9202803, -1.7600503, 2.8037138, -3.2643173, 2.6803305
2: -0.3788169, 1.0198433, -1.7155523, 3.1846526, -3.5634689, 2.7353957
3: -0.8183979, 1.0579665, -2.2817159, 3.6491742, -4.4675713, 3.3396823
4: -0.6781206, 1.2412384, -2.6121106, 3.7078054, -4.3859262, 3.8533490

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0472941, upper bound: 1.0478360
time: 0.51 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 4

Time for candidate selection: 3.24 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507335, upper bound: 1.0492755
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507335, upper bound: 1.0492755
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1981437, 0.6735660, -1.2890360, 2.6080267, -2.8061705, 1.9626021
1: -0.4075772, 0.8701689, -1.7757368, 2.8224382, -3.2300153, 2.6459057
2: -0.3290880, 0.9651487, -1.7412896, 3.1970921, -3.5261800, 2.7064383
3: -0.7565911, 0.9793513, -2.2828507, 3.6829684, -4.4395590, 3.2622020
4: -0.6072877, 1.1647243, -2.6450953, 3.7347975, -4.3420854, 3.8098197

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 6

Time for candidate selection: 2.75 seconds

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507640, upper bound: 1.0492897
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507640, upper bound: 1.0495664
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.2620909, 0.7696464, -1.2890360, 2.6080267, -2.8701177, 2.0586824
1: -0.5045343, 0.9919055, -1.7757368, 2.8224382, -3.3269725, 2.7676423
2: -0.4200620, 1.0960678, -1.7412896, 3.1970921, -3.6171536, 2.8373575
3: -0.8876588, 1.1395187, -2.2828507, 3.6829684, -4.5706272, 3.4223695
4: -0.7470124, 1.3261604, -2.6450953, 3.7347975, -4.4818096, 3.9712558

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 6

Time for candidate selection: 2.79 seconds

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507640, upper bound: 1.0493983
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507640, upper bound: 1.0496745
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1981437, 0.6735660, -1.3504868, 2.6251640, -2.8233078, 2.0240529
1: -0.4075772, 0.8701689, -1.8179705, 2.8394029, -3.2469802, 2.6881394
2: -0.3290880, 0.9651487, -1.7806034, 3.2355962, -3.5646842, 2.7457521
3: -0.7565911, 0.9793513, -2.3383503, 3.7266910, -4.4832807, 3.3177016
4: -0.6072877, 1.1647243, -2.7018437, 3.7814958, -4.3887835, 3.8665681

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0500000, mid=0.0500000, abs_max=1.1883488893508911
rel_dist={0: [-1.0553818619849304, 1.0553818619849311]}

## Binary search (step 2) starts
Candidate diff: 0.0250000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

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
- Time for IS candidates: 0.85 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.85
Output dim: 0, lower bound: -1.0531974, upper bound: 1.0509638
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.85
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.2352601, 0.7538407, -0.3035725, 0.8847764, -1.1200366, 1.0574131
1: -0.4706453, 0.9488738, -0.5660125, 1.0933844, -1.5640295, 1.5148864
2: -0.3915833, 1.0723588, -0.4826685, 1.2412479, -1.6328310, 1.5550274
3: -0.8320177, 1.0878556, -0.9617165, 1.2755736, -2.1075912, 2.0495720
4: -0.7029035, 1.3031529, -0.8354526, 1.4994075, -2.2023110, 2.1386056

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
time: 0.38 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
time: 0.36 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -1.3271873, 2.7002649, -0.3000076, 0.8778248, -2.2050121, 3.0002723
1: -1.8266034, 2.9257355, -0.5613543, 1.0854805, -2.9120839, 3.4870896
2: -1.7866864, 3.3100519, -0.4781591, 1.2323816, -3.0190680, 3.7882106
3: -2.3538351, 3.8103127, -0.9558605, 1.2658806, -3.6197157, 4.7661729
4: -2.7129741, 3.8588223, -0.8289866, 1.4898851, -4.2028589, 4.6878090

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
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
time: 0.33 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.30 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.30
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.30
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.30
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.30
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.2352601, 0.7538407, -0.2352601, 0.7538407, -0.9891008, 0.9891006
1: -0.4706453, 0.9488738, -0.4706453, 0.9488738, -1.4195192, 1.4195192
2: -0.3915833, 1.0723588, -0.3915833, 1.0723588, -1.4639422, 1.4639422
3: -0.8320177, 1.0878556, -0.8320177, 1.0878556, -1.9198732, 1.9198732
4: -0.7029035, 1.3031529, -0.7029035, 1.3031529, -2.0060563, 2.0060563

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27

Time for candidate selection: 0.12 seconds

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
time: 0.39 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.2352601, 0.7538407, -1.3271873, 2.7002649, -2.9355247, 2.0810280
1: -0.4706453, 0.9488738, -1.8266034, 2.9257355, -3.3963809, 2.7754772
2: -0.3915833, 1.0723588, -1.7866864, 3.3100519, -3.7016354, 2.8590453
3: -0.8320177, 1.0878556, -2.3538351, 3.8103127, -4.6423302, 3.4416907
4: -0.7029035, 1.3031529, -2.7129741, 3.8588223, -4.5617256, 4.0161266

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0527347, upper bound: 1.0502364
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0526130, upper bound: 1.0508594
time: 0.40 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -1.3271873, 2.7002649, -0.2352601, 0.7538407, -2.0810280, 2.9355249
1: -1.8266034, 2.9257355, -0.4706453, 0.9488738, -2.7754772, 3.3963809
2: -1.7866864, 3.3100519, -0.3915833, 1.0723588, -2.8590453, 3.7016351
3: -2.3538351, 3.8103127, -0.8320177, 1.0878556, -3.4416907, 4.6423302
4: -2.7129741, 3.8588223, -0.7029035, 1.3031529, -4.0161266, 4.5617256

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0481071, upper bound: 1.0485181
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0477081, upper bound: 1.0477081
time: 0.34 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -1.3271873, 2.7002649, -1.3271873, 2.7002649, -4.0274525, 4.0274525
1: -1.8266034, 2.9257355, -1.8266034, 2.9257355, -4.7523389, 4.7523384
2: -1.7866864, 3.3100519, -1.7866864, 3.3100519, -5.0967383, 5.0967379
3: -2.3538351, 3.8103127, -2.3538351, 3.8103127, -6.1641479, 6.1641479
4: -2.7129741, 3.8588223, -2.7129741, 3.8588223, -6.5717964, 6.5717964

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0481071, upper bound: 1.0485181
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0477081, upper bound: 1.0477081
time: 0.36 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.36 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.36
Output dim: 0, lower bound: -1.0527347, upper bound: 1.0502364
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.36
Output dim: 0, lower bound: -1.0526130, upper bound: 1.0508594
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.36
Output dim: 0, lower bound: -1.0527347, upper bound: 1.0502364
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.36
Output dim: 0, lower bound: -1.0526130, upper bound: 1.0508594
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 2.36
Output dim: 0, lower bound: -1.0481071, upper bound: 1.0485181
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.36
Output dim: 0, lower bound: -1.0477081, upper bound: 1.0477081
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 2.36
Output dim: 0, lower bound: -1.0481071, upper bound: 1.0485181
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.36
Output dim: 0, lower bound: -1.0477081, upper bound: 1.0477081

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.2013956, 0.7023641, -0.2201980, 0.7295994, -0.9309949, 0.9225621
1: -0.4206367, 0.8831170, -0.4483676, 0.9182405, -1.3388772, 1.3314846
2: -0.3466128, 1.0003821, -0.3716945, 1.0391469, -1.3857597, 1.3720765
3: -0.7602279, 1.0080743, -0.8000128, 1.0511825, -1.8114104, 1.8080871
4: -0.6370696, 1.2147777, -0.6739963, 1.2631135, -1.9001831, 1.8887740

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539857, upper bound: 1.0539857
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539857, upper bound: 1.0539857
time: 0.36 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.2363607, 0.7416377, -0.2207971, 0.7253933, -0.9617540, 0.9624348
1: -0.4722191, 0.9222932, -0.4493197, 0.9112105, -1.3834295, 1.3716129
2: -0.3964722, 1.0577438, -0.3732101, 1.0342234, -1.4306957, 1.4309539
3: -0.8244337, 1.0810699, -0.8019392, 1.0477722, -1.8722059, 1.8830092
4: -0.7141775, 1.3009543, -0.6769375, 1.2616144, -1.9757919, 1.9778919

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

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
0: -0.2013956, 0.7023641, -1.3166026, 2.6747036, -2.8760991, 2.0189667
1: -0.4206367, 0.8831170, -1.8116517, 2.8957775, -3.3164141, 2.6947687
2: -0.3466128, 1.0003821, -1.7737854, 3.2782125, -3.6248252, 2.7741675
3: -0.7602279, 1.0080743, -2.3325071, 3.7746639, -4.5348921, 3.3405814
4: -0.6370696, 1.2147777, -2.6938512, 3.8235836, -4.4606533, 3.9086289

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0527301, upper bound: 1.0495914
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0460222, upper bound: 1.0435204
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.2363607, 0.7416377, -1.3012817, 2.6305966, -2.8669572, 2.0429194
1: -0.4722191, 0.9222932, -1.7898400, 2.8466678, -3.3188868, 2.7121332
2: -0.3964722, 1.0577438, -1.7561550, 3.2289431, -3.6254153, 2.8138988
3: -0.8244337, 1.0810699, -2.3022947, 3.7166531, -4.5410867, 3.3833647
4: -0.7141775, 1.3009543, -2.6689839, 3.7697055, -4.4838829, 3.9699383

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513845, upper bound: 1.0495972
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513845, upper bound: 1.0508594
time: 0.43 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.90 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 0, lower bound: -1.0539857, upper bound: 1.0539857
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 0, lower bound: -1.0539857, upper bound: 1.0539857
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 0, lower bound: -1.0539857, upper bound: 1.0540170
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 0, lower bound: -1.0539857, upper bound: 1.0540170
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 0, lower bound: -1.0527301, upper bound: 1.0495914
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.90
Output dim: 0, lower bound: -1.0460222, upper bound: 1.0435204
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 0, lower bound: -1.0513845, upper bound: 1.0495972
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 0, lower bound: -1.0513845, upper bound: 1.0508594

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.2013956, 0.7023641, -0.2013956, 0.7023641, -0.9037597, 0.9037597
1: -0.4206367, 0.8831170, -0.4206367, 0.8831170, -1.3037536, 1.3037536
2: -0.3466128, 1.0003821, -0.3466128, 1.0003821, -1.3469949, 1.3469949
3: -0.7602279, 1.0080743, -0.7602279, 1.0080743, -1.7683022, 1.7683022
4: -0.6370696, 1.2147777, -0.6370696, 1.2147777, -1.8518473, 1.8518473

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0517957, upper bound: 1.0505564
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502114, upper bound: 1.0502114
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.2013956, 0.7023641, -0.2363607, 0.7416377, -0.9430333, 0.9387248
1: -0.4206367, 0.8831170, -0.4722191, 0.9222932, -1.3429298, 1.3553361
2: -0.3466128, 1.0003821, -0.3964722, 1.0577438, -1.4043566, 1.3968543
3: -0.7602279, 1.0080743, -0.8244337, 1.0810699, -1.8412979, 1.8325080
4: -0.6370696, 1.2147777, -0.7141775, 1.3009543, -1.9380239, 1.9289553

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0517957, upper bound: 1.0511818
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502114, upper bound: 1.0506232
time: 0.44 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.2363607, 0.7416377, -0.2013956, 0.7023641, -0.9387248, 0.9430333
1: -0.4722191, 0.9222932, -0.4206367, 0.8831170, -1.3553361, 1.3429298
2: -0.3964722, 1.0577438, -0.3466128, 1.0003821, -1.3968543, 1.4043566
3: -0.8244337, 1.0810699, -0.7602279, 1.0080743, -1.8325080, 1.8412979
4: -0.7141775, 1.3009543, -0.6370696, 1.2147777, -1.9289553, 1.9380239

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.2363607, 0.7416377, -0.2363607, 0.7416377, -0.9779984, 0.9779984
1: -0.4722191, 0.9222932, -0.4722191, 0.9222932, -1.3945123, 1.3945123
2: -0.3964722, 1.0577438, -0.3964722, 1.0577438, -1.4542160, 1.4542160
3: -0.8244337, 1.0810699, -0.8244337, 1.0810699, -1.9055036, 1.9055036
4: -0.7141775, 1.3009543, -0.7141775, 1.3009543, -2.0151320, 2.0151320

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1771595, 0.6718225, -1.2185290, 2.4550967, -2.6322563, 1.8903515
1: -0.3856640, 0.8468491, -1.6880636, 2.6817410, -3.0674045, 2.5349128
2: -0.3121465, 0.9535633, -1.6516476, 3.0208831, -3.3330293, 2.6052108
3: -0.7178994, 0.9569075, -2.1747322, 3.4732447, -4.1911440, 3.1316397
4: -0.5855403, 1.1548326, -2.5129638, 3.5335193, -4.1190596, 3.6677964

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0516232, upper bound: 1.0491312
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0497901, upper bound: 1.0482710
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.2326994, 0.7354137, -1.2320198, 2.5446754, -2.7773747, 1.9674335
1: -0.4671534, 0.9142381, -1.7103772, 2.7687185, -3.2358718, 2.6246152
2: -0.3918004, 1.0477358, -1.6705413, 3.1301744, -3.5219748, 2.7182770
3: -0.8169378, 1.0718229, -2.2186384, 3.5843155, -4.4012532, 3.2904613
4: -0.7069051, 1.2899147, -2.5484686, 3.6438389, -4.3507442, 3.8383832

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513845, upper bound: 1.0495972
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513845, upper bound: 1.0495972
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.2333232, 0.7381012, -1.2878053, 2.5980442, -2.8313670, 2.0259066
1: -0.4680505, 0.9182228, -1.7738810, 2.8132265, -3.2812769, 2.6921039
2: -0.3921940, 1.0523577, -1.7409041, 3.1906319, -3.5828259, 2.7932618
3: -0.8195140, 1.0749243, -2.2811434, 3.6725526, -4.4920664, 3.3560677
4: -0.7077722, 1.2939336, -2.6458459, 3.7281470, -4.4359188, 3.9397795

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0523143, upper bound: 1.0507842
time: 0.44 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0514137, upper bound: 1.0506707
time: 0.40 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.52 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 0, lower bound: -1.0517957, upper bound: 1.0505564
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 0, lower bound: -1.0502114, upper bound: 1.0502114
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 0, lower bound: -1.0517957, upper bound: 1.0511818
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 0, lower bound: -1.0502114, upper bound: 1.0506232
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 0, lower bound: -1.0516232, upper bound: 1.0491312
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 0, lower bound: -1.0497901, upper bound: 1.0482710
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 0, lower bound: -1.0513845, upper bound: 1.0495972
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 0, lower bound: -1.0513845, upper bound: 1.0495972
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 0, lower bound: -1.0523143, upper bound: 1.0507842
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.52
Output dim: 0, lower bound: -1.0514137, upper bound: 1.0506707

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1851257, 0.6783313, -0.2013956, 0.7023641, -0.8874898, 0.8797269
1: -0.3969364, 0.8526834, -0.4206367, 0.8831170, -1.2800534, 1.2733200
2: -0.3242711, 0.9644121, -0.3466128, 1.0003821, -1.3246531, 1.3110249
3: -0.7251614, 0.9704387, -0.7602279, 1.0080743, -1.7332357, 1.7306666
4: -0.6038694, 1.1718525, -0.6370696, 1.2147777, -1.8186471, 1.8089221

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502114, upper bound: 1.0502114
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502114, upper bound: 1.0502114
time: 0.50 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.2674521, 0.8277271, -0.1712943, 0.6526494, -0.9201015, 0.9990214
1: -0.5075184, 1.0548140, -0.3726498, 0.8235356, -1.3310540, 1.4274638
2: -0.4327361, 1.2001319, -0.3017510, 0.9336209, -1.3663570, 1.5018828
3: -0.9017408, 1.1985904, -0.7055027, 0.9300249, -1.8317657, 1.9040930
4: -0.7864016, 1.4086894, -0.5726804, 1.1268942, -1.9132959, 1.9813697

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502114, upper bound: 1.0502114
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502114, upper bound: 1.0502114
time: 0.36 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1851257, 0.6783313, -0.2363607, 0.7416377, -0.9267634, 0.9146920
1: -0.3969364, 0.8526834, -0.4722191, 0.9222932, -1.3192296, 1.3249025
2: -0.3242711, 0.9644121, -0.3964722, 1.0577438, -1.3820149, 1.3608843
3: -0.7251614, 0.9704387, -0.8244337, 1.0810699, -1.8062314, 1.7948724
4: -0.6038694, 1.1718525, -0.7141775, 1.3009543, -1.9048238, 1.8860300

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502114, upper bound: 1.0506232
time: 0.47 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502114, upper bound: 1.0506232
time: 0.44 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.2674521, 0.8277271, -0.2106718, 0.6987747, -0.9662268, 1.0383989
1: -0.5075184, 1.0548140, -0.4317660, 0.8731946, -1.3807130, 1.4865800
2: -0.4327361, 1.2001319, -0.3575066, 1.0027628, -1.4354990, 1.5576384
3: -0.9017408, 1.1985904, -0.7800540, 1.0120336, -1.9137744, 1.9786444
4: -0.7864016, 1.4086894, -0.6589954, 1.2239647, -2.0103664, 2.0676849

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504534, upper bound: 1.0506232
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502114, upper bound: 1.0506232
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.2631326, 0.7698788, -0.1973387, 0.6954002, -0.9585328, 0.9672174
1: -0.5045100, 0.9689885, -0.4149089, 0.8743680, -1.3788780, 1.3838973
2: -0.4292829, 1.0956014, -0.3412917, 0.9895573, -1.4188402, 1.4368931
3: -0.8731403, 1.1296451, -0.7518593, 0.9971166, -1.8702569, 1.8815044
4: -0.7632787, 1.3396218, -0.6289587, 1.2026020, -1.9658808, 1.9685805

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0544350
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0544350
time: 0.35 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.2254990, 0.7289720, -0.1985539, 0.6989882, -0.9244872, 0.9275258
1: -0.4573010, 0.9075529, -0.4167078, 0.8790563, -1.3363574, 1.3242607
2: -0.3811812, 1.0387152, -0.3425781, 0.9952359, -1.3764172, 1.3812933
3: -0.8068940, 1.0590353, -0.7554929, 1.0022113, -1.8091054, 1.8145282
4: -0.6912529, 1.2759079, -0.6310958, 1.2082970, -1.8995500, 1.9070036

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0511818, upper bound: 1.0521688
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506232, upper bound: 1.0504534
time: 0.40 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.2631326, 0.7698788, -0.2326994, 0.7354137, -0.9985462, 1.0025783
1: -0.5045100, 0.9689885, -0.4671534, 0.9142381, -1.4187481, 1.4361420
2: -0.4292829, 1.0956014, -0.3918004, 1.0477358, -1.4770187, 1.4874018
3: -0.8731403, 1.1296451, -0.8169378, 1.0718229, -1.9449632, 1.9465829
4: -0.7632787, 1.3396218, -0.7069051, 1.2899147, -2.0531936, 2.0465269

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.2254990, 0.7289720, -0.2333232, 0.7381012, -0.9636002, 0.9622952
1: -0.4573010, 0.9075529, -0.4680505, 0.9182228, -1.3755238, 1.3756034
2: -0.3811812, 1.0387152, -0.3921940, 1.0523577, -1.4335389, 1.4309093
3: -0.8068940, 1.0590353, -0.8195140, 1.0749243, -1.8818183, 1.8785493
4: -0.6912529, 1.2759079, -0.7077722, 1.2939336, -1.9851866, 1.9836800

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0515796, upper bound: 1.0523694
time: 0.49 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513563, upper bound: 1.0514759
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1598512, 0.6470749, -1.2185290, 2.4550967, -2.6149478, 1.8656039
1: -0.3604083, 0.8156391, -1.6880636, 2.6817410, -3.0421491, 2.5037026
2: -0.2882156, 0.9171838, -1.6516476, 3.0208831, -3.3090982, 2.5688314
3: -0.6813787, 0.9178867, -2.1747322, 3.4732447, -4.1546230, 3.0926189
4: -0.5500027, 1.1105816, -2.5129638, 3.5335193, -4.0835218, 3.6235454

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0497181, upper bound: 1.0482710
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0497181, upper bound: 1.0482710
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.2082972, 0.7256089, -1.1202302, 2.2472436, -2.4555404, 1.8458391
1: -0.4265601, 0.9231466, -1.5566983, 2.4561176, -2.8826776, 2.4798450
2: -0.3480972, 1.0395398, -1.5275354, 2.7807105, -3.1288075, 2.5670753
3: -0.7874584, 1.0466194, -2.0135312, 3.1839292, -3.9713876, 3.0601506
4: -0.6579083, 1.2393422, -2.3365993, 3.2605400, -3.9184484, 3.5759416

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0497181, upper bound: 1.0482710
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0497181, upper bound: 1.0482710
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.2631326, 0.7698788, -1.2320198, 2.5446754, -2.8078079, 2.0018985
1: -0.5045100, 0.9689885, -1.7103772, 2.7687185, -3.2732286, 2.6793656
2: -0.4292829, 1.0956014, -1.6705413, 3.1301744, -3.5594573, 2.7661428
3: -0.8731403, 1.1296451, -2.2186384, 3.5843155, -4.4574556, 3.3482835
4: -0.7632787, 1.3396218, -2.5484686, 3.6438389, -4.4071169, 3.8880904

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0495441
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0495972
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.2254990, 0.7289720, -1.2320198, 2.5446754, -2.7701743, 1.9609917
1: -0.4573010, 0.9075529, -1.7103772, 2.7687185, -3.2260194, 2.6179302
2: -0.3811812, 1.0387152, -1.6705413, 3.1301744, -3.5113554, 2.7092566
3: -0.8068940, 1.0590353, -2.2186384, 3.5843155, -4.3912096, 3.2776737
4: -0.6912529, 1.2759079, -2.5484686, 3.6438389, -4.3350916, 3.8243766

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0495441
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0495972
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.2092665, 0.7062556, -1.2878053, 2.5980442, -2.8073106, 1.9940609
1: -0.4340345, 0.8779389, -1.7738810, 2.8132265, -3.2472610, 2.6518197
2: -0.3589701, 1.0076040, -1.7409041, 3.1906319, -3.5496020, 2.7485080
3: -0.7731280, 1.0230377, -2.2811434, 3.6725526, -4.4456806, 3.3041811
4: -0.6592926, 1.2364941, -2.6458459, 3.7281470, -4.3874397, 3.8823400

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0514137, upper bound: 1.0506707
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0514137, upper bound: 1.0506707
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.2990957, 0.8537487, -1.1989313, 2.4155779, -2.7146735, 2.0526800
1: -0.5549765, 1.0752316, -1.6564941, 2.6148190, -3.1697950, 2.7317257
2: -0.4782404, 1.2304351, -1.6270294, 2.9819658, -3.4602060, 2.8574646
3: -0.9541719, 1.2576739, -2.1421685, 3.4155142, -4.3696861, 3.3998423
4: -0.8555491, 1.4712769, -2.4846714, 3.4886141, -4.3441629, 3.9559484

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0514137, upper bound: 1.0506707
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0514137, upper bound: 1.0506707
time: 0.37 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.58 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -1.0502114, upper bound: 1.0502114
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -1.0502114, upper bound: 1.0502114
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -1.0502114, upper bound: 1.0502114
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -1.0502114, upper bound: 1.0502114
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -1.0502114, upper bound: 1.0506232
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -1.0502114, upper bound: 1.0506232
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -1.0504534, upper bound: 1.0506232
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -1.0502114, upper bound: 1.0506232
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0544350
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0544350
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -1.0511818, upper bound: 1.0521688
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -1.0506232, upper bound: 1.0504534
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -1.0515796, upper bound: 1.0523694
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -1.0513563, upper bound: 1.0514759
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -1.0497181, upper bound: 1.0482710
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -1.0497181, upper bound: 1.0482710
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -1.0497181, upper bound: 1.0482710
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -1.0497181, upper bound: 1.0482710
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0495441
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0495972
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0495441
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0495972
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -1.0514137, upper bound: 1.0506707
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -1.0514137, upper bound: 1.0506707
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -1.0514137, upper bound: 1.0506707
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 0, lower bound: -1.0514137, upper bound: 1.0506707

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1851257, 0.6783313, -0.1851257, 0.6783313, -0.8634571, 0.8634571
1: -0.3969364, 0.8526834, -0.3969364, 0.8526834, -1.2496197, 1.2496197
2: -0.3242711, 0.9644121, -0.3242711, 0.9644121, -1.2886832, 1.2886832
3: -0.7251614, 0.9704387, -0.7251614, 0.9704387, -1.6956002, 1.6956002
4: -0.6038694, 1.1718525, -0.6038694, 1.1718525, -1.7757219, 1.7757219

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0515718, upper bound: 1.0504426
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0515718, upper bound: 1.0503786
time: 0.43 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1851257, 0.6783313, -0.2674521, 0.8277271, -1.0128528, 0.9457834
1: -0.3969364, 0.8526834, -0.5075184, 1.0548140, -1.4517504, 1.3602018
2: -0.3242711, 0.9644121, -0.4327361, 1.2001319, -1.5244030, 1.3971481
3: -0.7251614, 0.9704387, -0.9017408, 1.1985904, -1.9237518, 1.8721795
4: -0.6038694, 1.1718525, -0.7864016, 1.4086894, -2.0125589, 1.9582541

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0515718, upper bound: 1.0504426
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0515718, upper bound: 1.0503786
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.2674521, 0.8277271, -0.1757940, 0.6660366, -0.9334887, 1.0035211
1: -0.5075184, 1.0548140, -0.3831117, 0.8383465, -1.3458649, 1.4379257
2: -0.4327361, 1.2001319, -0.3106660, 0.9465988, -1.3793349, 1.5107979
3: -0.9017408, 1.1985904, -0.7074553, 0.9505194, -1.8522602, 1.9060457
4: -0.7864016, 1.4086894, -0.5832559, 1.1489482, -1.9353498, 1.9919453

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500954, upper bound: 1.0501320
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500529, upper bound: 1.0500529
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.2674521, 0.8277271, -0.2548658, 0.7943116, -1.0617638, 1.0825928
1: -0.5075184, 1.0548140, -0.4891222, 1.0074059, -1.5149243, 1.5439363
2: -0.4327361, 1.2001319, -0.4157726, 1.1434062, -1.5761423, 1.6159046
3: -0.9017408, 1.1985904, -0.8697562, 1.1547513, -2.0564921, 2.0683465
4: -0.7864016, 1.4086894, -0.7592028, 1.3576846, -2.1440864, 2.1678922

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500954, upper bound: 1.0501320
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500529, upper bound: 1.0500529
time: 0.44 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1851257, 0.6783313, -0.2122585, 0.7096910, -0.8948168, 0.8905898
1: -0.3969364, 0.8526834, -0.4381096, 0.8818882, -1.2788246, 1.2907929
2: -0.3242711, 0.9644121, -0.3631817, 1.0126708, -1.3369418, 1.3275938
3: -0.7251614, 0.9704387, -0.7779223, 1.0290604, -1.7542218, 1.7483611
4: -0.6038694, 1.1718525, -0.6656082, 1.2431200, -1.8469894, 1.8374606

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 4

Time for candidate selection: 2.53 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0497071, upper bound: 1.0499520
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0520851, upper bound: 1.0510982
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1851257, 0.6783313, -0.3022564, 0.8574660, -1.0425918, 0.9805877
1: -0.3969364, 0.8526834, -0.5588838, 1.0796481, -1.4765846, 1.4115672
2: -0.3242711, 0.9644121, -0.4826456, 1.2362378, -1.5605088, 1.4470577
3: -0.7251614, 0.9704387, -0.9591278, 1.2640674, -1.9892288, 1.9295666
4: -0.6038694, 1.1718525, -0.8620855, 1.4785453, -2.0824146, 2.0339379

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 4

Time for candidate selection: 2.54 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0497071, upper bound: 1.0499520
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0516988, upper bound: 1.0510983
time: 0.47 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.2674521, 0.8277271, -0.2122585, 0.7096910, -0.9771431, 1.0399857
1: -0.5075184, 1.0548140, -0.4381096, 0.8818882, -1.3894066, 1.4929236
2: -0.4327361, 1.2001319, -0.3631817, 1.0126708, -1.4454069, 1.5633136
3: -0.9017408, 1.1985904, -0.7779223, 1.0290604, -1.9308012, 1.9765127
4: -0.7864016, 1.4086894, -0.6656082, 1.2431200, -2.0295215, 2.0742974

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 6

Time for candidate selection: 2.45 seconds

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0490002, upper bound: 1.0480409
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0458380, upper bound: 1.0456311
time: 0.45 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.2674521, 0.8277271, -0.3022564, 0.8574660, -1.1249182, 1.1299834
1: -0.5075184, 1.0548140, -0.5588838, 1.0796481, -1.5871665, 1.6136978
2: -0.4327361, 1.2001319, -0.4826456, 1.2362378, -1.6689739, 1.6827774
3: -0.9017408, 1.1985904, -0.9591278, 1.2640674, -2.1658082, 2.1577182
4: -0.7864016, 1.4086894, -0.8620855, 1.4785453, -2.2649469, 2.2707748

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
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
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 6

Time for candidate selection: 2.48 seconds

### Candidate
type: A, layer: 3, pos: 17

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0490002, upper bound: 1.0480409
time: 0.45 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0434735, upper bound: 1.0441066
time: 0.47 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.2631326, 0.7698788, -0.2350464, 0.7415904, -1.0047231, 1.0049253
1: -0.5045100, 0.9689885, -0.4636297, 0.9372106, -1.4417207, 1.4326181
2: -0.4292829, 1.0956014, -0.3912313, 1.0568323, -1.4861152, 1.4868327
3: -0.8731403, 1.1296451, -0.8212245, 1.0728223, -1.9459627, 1.9508696
4: -0.7632787, 1.3396218, -0.7068326, 1.2774920, -2.0407708, 2.0464544

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0537060, upper bound: 1.0543926
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0528018, upper bound: 1.0532212
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 4

Time for candidate selection: 3.30 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0540642
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0541033
time: 0.41 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.2631326, 0.7698788, -0.1909873, 0.6899648, -0.9530973, 0.9608661
1: -0.5045100, 0.9689885, -0.4062611, 0.8681383, -1.3726482, 1.3752496
2: -0.4292829, 1.0956014, -0.3319388, 0.9815091, -1.4107921, 1.4275403
3: -0.8731403, 1.1296451, -0.7429183, 0.9865521, -1.8596925, 1.8725634
4: -0.7632787, 1.3396218, -0.6151925, 1.1911318, -1.9544106, 1.9548143

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0537060, upper bound: 1.0543926
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0528018, upper bound: 1.0532212
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 4

Time for candidate selection: 3.31 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0540642
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0541033
time: 0.38 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.2254990, 0.7289720, -0.1822524, 0.6750042, -0.9005032, 0.9112244
1: -0.4573010, 0.9075529, -0.3929973, 0.8486877, -1.3059888, 1.3005502
2: -0.3811812, 1.0387152, -0.3201882, 0.9593878, -1.3405690, 1.3589034
3: -0.8068940, 1.0590353, -0.7204593, 0.9646156, -1.7715096, 1.7794945
4: -0.6912529, 1.2759079, -0.5978308, 1.1654682, -1.8567212, 1.8737386

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506232, upper bound: 1.0504534
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506232, upper bound: 1.0504534
time: 0.42 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.1994102, 0.6853333, -0.2640961, 0.8231158, -1.0225260, 0.9494294
1: -0.4165157, 0.8576184, -0.5030945, 1.0491726, -1.4656883, 1.3607129
2: -0.3407917, 0.9829090, -0.4279521, 1.1928697, -1.5336614, 1.4108611
3: -0.7614180, 0.9885427, -0.8960606, 1.1909752, -1.9523932, 1.8846033
4: -0.6340601, 1.1983819, -0.7792837, 1.4003186, -2.0343788, 1.9776657

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506232, upper bound: 1.0504534
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506232, upper bound: 1.0504534
time: 0.46 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.2631326, 0.7698788, -0.2631326, 0.7698788, -1.0330114, 1.0330114
1: -0.5045100, 0.9689885, -0.5045100, 0.9689885, -1.4734986, 1.4734986
2: -0.4292829, 1.0956014, -0.4292829, 1.0956014, -1.5248843, 1.5248843
3: -0.8731403, 1.1296451, -0.8731403, 1.1296451, -2.0027854, 2.0027854
4: -0.7632787, 1.3396218, -0.7632787, 1.3396218, -2.1029005, 2.1029005

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 4

Time for candidate selection: 2.56 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0536722
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0537102
time: 0.37 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2631326, 0.7698788, -0.2254990, 0.7289720, -0.9921045, 0.9953778
1: -0.5045100, 0.9689885, -0.4573010, 0.9075529, -1.4120629, 1.4262896
2: -0.4292829, 1.0956014, -0.3811812, 1.0387152, -1.4679981, 1.4767827
3: -0.8731403, 1.1296451, -0.8068940, 1.0590353, -1.9321756, 1.9365392
4: -0.7632787, 1.3396218, -0.6912529, 1.2759079, -2.0391865, 2.0308747

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 4

Time for candidate selection: 2.64 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0536722
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0537102
time: 0.40 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.2254990, 0.7289720, -0.2092665, 0.7062556, -0.9317545, 0.9382384
1: -0.4573010, 0.9075529, -0.4340345, 0.8779389, -1.3352399, 1.3415873
2: -0.3811812, 1.0387152, -0.3589701, 1.0076040, -1.3887852, 1.3976853
3: -0.8068940, 1.0590353, -0.7731280, 1.0230377, -1.8299317, 1.8321633
4: -0.6912529, 1.2759079, -0.6592926, 1.2364941, -1.9277470, 1.9352005

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513563, upper bound: 1.0514759
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513563, upper bound: 1.0514759
time: 0.47 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.1994102, 0.6853333, -0.2990957, 0.8537487, -1.0531589, 0.9844290
1: -0.4165157, 0.8576184, -0.5549765, 1.0752316, -1.4917473, 1.4125948
2: -0.3407917, 0.9829090, -0.4782404, 1.2304351, -1.5712268, 1.4611495
3: -0.7614180, 0.9885427, -0.9541719, 1.2576739, -2.0190918, 1.9427146
4: -0.6340601, 1.1983819, -0.8555491, 1.4712769, -2.1053371, 2.0539310

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513563, upper bound: 1.0514759
time: 0.45 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513563, upper bound: 1.0514759
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1598512, 0.6470749, -1.2048244, 2.4248323, -2.5846834, 1.8518994
1: -0.3604083, 0.8156391, -1.6693063, 2.6460402, -3.0064485, 2.4849453
2: -0.2882156, 0.9171838, -1.6358550, 2.9823542, -3.2705698, 2.5530386
3: -0.6813787, 0.9178867, -2.1484208, 3.4295993, -4.1109781, 3.0663075
4: -0.5500027, 1.1105816, -2.4897370, 3.4908578, -4.0408602, 3.6003187

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0516232, upper bound: 1.0491312
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0516232, upper bound: 1.0491312
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1598512, 0.6470749, -1.1359986, 2.3385446, -2.4983954, 1.7830735
1: -0.3604083, 0.8156391, -1.5716488, 2.5924485, -2.9528563, 2.3872881
2: -0.2882156, 0.9171838, -1.5189495, 2.9331117, -3.2213271, 2.4361334
3: -0.6813787, 0.9178867, -2.0836442, 3.2931526, -3.9745309, 3.0015309
4: -0.5500027, 1.1105816, -2.3410478, 3.3857694, -3.9357719, 3.4516294

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0516232, upper bound: 1.0491312
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0516232, upper bound: 1.0491312
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.2082972, 0.7256089, -1.1868396, 2.3870296, -2.5953269, 1.9124485
1: -0.4265601, 0.9231466, -1.6473439, 2.6067681, -3.0333281, 2.5704904
2: -0.3480972, 1.0395398, -1.6157327, 2.9373336, -3.2854309, 2.6552725
3: -0.7874584, 1.0466194, -2.1182137, 3.3775549, -4.1650133, 3.1648331
4: -0.6579083, 1.2393422, -2.4597297, 3.4410822, -4.0989904, 3.6990719

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0497181, upper bound: 1.0482710
time: 0.44 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0497181, upper bound: 1.0482710
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.2082972, 0.7256089, -1.1359986, 2.3385446, -2.5468414, 1.8616076
1: -0.4265601, 0.9231466, -1.5716488, 2.5924485, -3.0190086, 2.4947953
2: -0.3480972, 1.0395398, -1.5189495, 2.9331117, -3.2812088, 2.5584893
3: -0.7874584, 1.0466194, -2.0836442, 3.2931526, -4.0806108, 3.1302636
4: -0.6579083, 1.2393422, -2.3410478, 3.3857694, -4.0436773, 3.5803900

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0497181, upper bound: 1.0482710
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0497181, upper bound: 1.0482710
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.2631326, 0.7698788, -1.2317941, 2.5485647, -2.8116972, 2.0016730
1: -0.5045100, 0.9689885, -1.7097392, 2.7696524, -3.2741623, 2.6787276
2: -0.4292829, 1.0956014, -1.6688566, 3.1283512, -3.5576341, 2.7644582
3: -0.8731403, 1.1296451, -2.2169631, 3.5870750, -4.4602156, 3.3466082
4: -0.7632787, 1.3396218, -2.5448000, 3.6422343, -4.4055128, 3.8844218

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 4

Time for candidate selection: 2.52 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0491687
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0491653
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.2631326, 0.7698788, -1.3011413, 2.5799091, -2.8430414, 2.0710201
1: -0.5045100, 0.9689885, -1.7600503, 2.8037138, -3.3082237, 2.7290387
2: -0.4292829, 1.0956014, -1.7155523, 3.1846526, -3.6139355, 2.8111539
3: -0.8731403, 1.1296451, -2.2817159, 3.6491742, -4.5223141, 3.4113610
4: -0.7632787, 1.3396218, -2.6121106, 3.7078054, -4.4710827, 3.9517324

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 4

Time for candidate selection: 2.67 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0492052
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0492247
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.2254990, 0.7289720, -1.2317941, 2.5485647, -2.7740636, 1.9607661
1: -0.4573010, 0.9075529, -1.7097392, 2.7696524, -3.2269535, 2.6172922
2: -0.3811812, 1.0387152, -1.6688566, 3.1283512, -3.5095320, 2.7075720
3: -0.8068940, 1.0590353, -2.2169631, 3.5870750, -4.3939691, 3.2759984
4: -0.6912529, 1.2759079, -2.5448000, 3.6422343, -4.3334870, 3.8207078

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 4

Time for candidate selection: 2.77 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0491257
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0491712
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.2254990, 0.7289720, -1.3011413, 2.5799091, -2.8054080, 2.0301132
1: -0.4573010, 0.9075529, -1.7600503, 2.8037138, -3.2610145, 2.6676033
2: -0.3811812, 1.0387152, -1.7155523, 3.1846526, -3.5658336, 2.7542677
3: -0.8068940, 1.0590353, -2.2817159, 3.6491742, -4.4560671, 3.3407512
4: -0.6912529, 1.2759079, -2.6121106, 3.7078054, -4.3990583, 3.8880186

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 4

Time for candidate selection: 2.59 seconds

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0491753
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0492310
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.2092665, 0.7062556, -1.2711267, 2.5596581, -2.7689242, 1.9773823
1: -0.4340345, 0.8779389, -1.7502675, 2.7682538, -3.2022882, 2.6282063
2: -0.3589701, 1.0076040, -1.7215834, 3.1407604, -3.4997303, 2.7291875
3: -0.7731280, 1.0230377, -2.2464452, 3.6179788, -4.3911066, 3.2694829
4: -0.6592926, 1.2364941, -2.6174710, 3.6723778, -4.3316703, 3.8539650

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0397093, upper bound: 1.0356131
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507505, upper bound: 1.0489614
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507007, upper bound: 1.0488390
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2092665, 0.7062556, -1.2713693, 2.6420059, -2.8512723, 1.9776249
1: -0.4340345, 0.8779389, -1.7414551, 2.9063215, -3.3403559, 2.6193938
2: -0.3589701, 1.0076040, -1.6821582, 3.2947881, -3.6537583, 2.6897621
3: -0.7731280, 1.0230377, -2.3091674, 3.7021747, -4.4753027, 3.3322051
4: -0.6592926, 1.2364941, -2.5841229, 3.7822459, -4.4415383, 3.8206170

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0397093, upper bound: 1.0435455
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507505, upper bound: 1.0489614
time: 0.46 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507007, upper bound: 1.0488390
time: 0.46 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.2990957, 0.8537487, -1.2711267, 2.5596581, -2.8587537, 2.1248755
1: -0.5549765, 1.0752316, -1.7502675, 2.7682538, -3.3232300, 2.8254991
2: -0.4782404, 1.2304351, -1.7215834, 3.1407604, -3.6190004, 2.9520185
3: -0.9541719, 1.2576739, -2.2464452, 3.6179788, -4.5721507, 3.5041189
4: -0.8555491, 1.4712769, -2.6174710, 3.6723778, -4.5279269, 4.0887480

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0390184, upper bound: 1.0355211
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501101, upper bound: 1.0488486
time: 0.49 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499329, upper bound: 1.0486672
time: 0.44 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.2990957, 0.8537487, -1.2713693, 2.6420059, -2.9411016, 2.1251180
1: -0.5549765, 1.0752316, -1.7414551, 2.9063215, -3.4612980, 2.8166866
2: -0.4782404, 1.2304351, -1.6821582, 3.2947881, -3.7730284, 2.9125934
3: -0.9541719, 1.2576739, -2.3091674, 3.7021747, -4.6563463, 3.5668411
4: -0.8555491, 1.4712769, -2.5841229, 3.7822459, -4.6377950, 4.0553999

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0390184, upper bound: 1.0434068
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501101, upper bound: 1.0488486
time: 0.44 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499329, upper bound: 1.0486672
time: 0.43 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.37 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0515718, upper bound: 1.0504426
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0515718, upper bound: 1.0503786
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0515718, upper bound: 1.0504426
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0515718, upper bound: 1.0503786
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0500954, upper bound: 1.0501320
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0500529, upper bound: 1.0500529
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0500954, upper bound: 1.0501320
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0500529, upper bound: 1.0500529
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0497071, upper bound: 1.0499520
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0520851, upper bound: 1.0510982
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0497071, upper bound: 1.0499520
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0516988, upper bound: 1.0510983
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0490002, upper bound: 1.0480409
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0458380, upper bound: 1.0456311
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0490002, upper bound: 1.0480409
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0434735, upper bound: 1.0441066
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0540642
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0541033
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0540642
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0541033
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0506232, upper bound: 1.0504534
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0506232, upper bound: 1.0504534
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0506232, upper bound: 1.0504534
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0506232, upper bound: 1.0504534
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0536722
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0537102
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0536722
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0536938, upper bound: 1.0537102
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0513563, upper bound: 1.0514759
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0513563, upper bound: 1.0514759
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0513563, upper bound: 1.0514759
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0513563, upper bound: 1.0514759
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0516232, upper bound: 1.0491312
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0516232, upper bound: 1.0491312
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0516232, upper bound: 1.0491312
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0516232, upper bound: 1.0491312
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0497181, upper bound: 1.0482710
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0497181, upper bound: 1.0482710
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0497181, upper bound: 1.0482710
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0497181, upper bound: 1.0482710
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0491687
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0491653
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0492052
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0492247
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0491257
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0491712
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0491753
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0508499, upper bound: 1.0492310
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0507505, upper bound: 1.0489614
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0507007, upper bound: 1.0488390
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0507505, upper bound: 1.0489614
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0507007, upper bound: 1.0488390
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0501101, upper bound: 1.0488486
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0499329, upper bound: 1.0486672
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0501101, upper bound: 1.0488486
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.37
Output dim: 0, lower bound: -1.0499329, upper bound: 1.0486672

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1476631, 0.6317905, -0.1598512, 0.6470749, -0.7947380, 0.7916417
1: -0.3481529, 0.7843536, -0.3604083, 0.8156391, -1.1637920, 1.1447619
2: -0.2732983, 0.8929532, -0.2882156, 0.9171838, -1.1904821, 1.1811688
3: -0.6720839, 0.8882244, -0.6813787, 0.9178867, -1.5899706, 1.5696031
4: -0.5259173, 1.0916876, -0.5500027, 1.1105816, -1.6364989, 1.6416903

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0542057, upper bound: 1.0547479
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0541776, upper bound: 1.0547296
time: 0.38 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1639505, 0.6451314, -0.1700749, 0.6564915, -0.8204420, 0.8152063
1: -0.3644042, 0.8125864, -0.3745029, 0.8262144, -1.1906186, 1.1870893
2: -0.2932469, 0.9175444, -0.3021806, 0.9340038, -1.2272507, 1.2197250
3: -0.6829748, 0.9167686, -0.6969874, 0.9343054, -1.6172802, 1.6137559
4: -0.5562484, 1.1130972, -0.5703892, 1.1335332, -1.6897817, 1.6834865

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0548057, upper bound: 1.0548266
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0541776, upper bound: 1.0548146
time: 0.44 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1476631, 0.6317905, -0.2266859, 0.7721168, -0.9197798, 0.8584764
1: -0.3481529, 0.7843536, -0.4527406, 0.9874769, -1.3356298, 1.2370942
2: -0.2732983, 0.8929532, -0.3731551, 1.1153061, -1.3886044, 1.2661083
3: -0.6720839, 0.8882244, -0.8304829, 1.1087748, -1.7808586, 1.7187073
4: -0.5259173, 1.0916876, -0.6979334, 1.3082323, -1.8341496, 1.7896209

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0515718, upper bound: 1.0503786
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0515718, upper bound: 1.0503786
time: 0.35 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1639505, 0.6451314, -0.2556306, 0.8070110, -0.9709615, 0.9007620
1: -0.3644042, 0.8125864, -0.4907007, 1.0281086, -1.3925128, 1.3032871
2: -0.2932469, 0.9175444, -0.4163584, 1.1684189, -1.4616657, 1.3339028
3: -0.6829748, 0.9167686, -0.8782174, 1.1671729, -1.8501477, 1.7949860
4: -0.5562484, 1.1130972, -0.7608104, 1.3744850, -1.9307334, 1.8739076

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0516646, upper bound: 1.0503786
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0516646, upper bound: 1.0503786
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1829112, 0.7200209, -0.1527726, 0.6375644, -0.8204756, 0.8727934
1: -0.3864989, 0.9190384, -0.3496050, 0.8043983, -1.1908972, 1.2686434
2: -0.3115277, 1.0421524, -0.2778672, 0.9032050, -1.2147328, 1.3200196
3: -0.7443157, 1.0138530, -0.6669772, 0.9020690, -1.6463847, 1.6808302
4: -0.6034802, 1.2099588, -0.5339428, 1.0919135, -1.6953937, 1.7439015

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0503786, upper bound: 1.0515718
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0503786, upper bound: 1.0516646
time: 0.40 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.2494289, 0.7944719, -0.1612186, 0.6450332, -0.8944621, 0.9556905
1: -0.4813044, 1.0119214, -0.3612567, 0.8129265, -1.2942309, 1.3731781
2: -0.4077803, 1.1496273, -0.2891539, 0.9176629, -1.3254433, 1.4387813
3: -0.8642187, 1.1482675, -0.6802358, 0.9154710, -1.7796896, 1.8285034
4: -0.7469358, 1.3539220, -0.5505019, 1.1120806, -1.8590164, 1.9044240

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0503786, upper bound: 1.0515718
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0503786, upper bound: 1.0516646
time: 0.40 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1829112, 0.7200209, -0.2164716, 0.7439526, -0.9268638, 0.9364924
1: -0.3864989, 0.9190384, -0.4377360, 0.9474247, -1.3339236, 1.3567743
2: -0.3115277, 1.0421524, -0.3592768, 1.0688229, -1.3803506, 1.4014292
3: -0.7443157, 1.0138530, -0.8040558, 1.0719622, -1.8162780, 1.8179088
4: -0.6034802, 1.2099588, -0.6755751, 1.2662123, -1.8696926, 1.8855338

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500529, upper bound: 1.0500529
time: 0.46 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500529, upper bound: 1.0500529
time: 0.44 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.2494289, 0.7944719, -0.2442495, 0.7768902, -1.0263190, 1.0387213
1: -0.4813044, 1.0119214, -0.4742511, 0.9854436, -1.4667480, 1.4861724
2: -0.4077803, 1.1496273, -0.4009653, 1.1176167, -1.5253969, 1.5505927
3: -0.8642187, 1.1482675, -0.8493451, 1.1276112, -1.9918299, 1.9976126
4: -0.7469358, 1.3539220, -0.7361979, 1.3283796, -2.0753155, 2.0901198

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500529, upper bound: 1.0500529
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500529, upper bound: 1.0500529
time: 0.41 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1280621, 0.5901273, -0.2122585, 0.7096910, -0.8377531, 0.8023859
1: -0.3120337, 0.7594209, -0.4381096, 0.8818882, -1.1939218, 1.1975305
2: -0.2382188, 0.8421853, -0.3631817, 1.0126708, -1.2508895, 1.2053671
3: -0.6190236, 0.8373086, -0.7779223, 1.0290604, -1.6480839, 1.6152310
4: -0.4718652, 1.0084939, -0.6656082, 1.2431200, -1.7149851, 1.6741021

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0541033, upper bound: 1.0535085
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0543909, upper bound: 1.0535085
time: 0.41 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.2275870, 0.7157916, -0.2122585, 0.7096910, -0.9372779, 0.9280502
1: -0.4526966, 0.9158092, -0.4381096, 0.8818882, -1.3345847, 1.3539188
2: -0.3724530, 1.0130844, -0.3631817, 1.0126708, -1.3851237, 1.3762661
3: -0.7981911, 1.0494591, -0.7779223, 1.0290604, -1.8272514, 1.8273814
4: -0.6660109, 1.2261832, -0.6656082, 1.2431200, -1.9091308, 1.8917913

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0541033, upper bound: 1.0536938
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0543967, upper bound: 1.0536938
time: 0.41 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1280621, 0.5901273, -0.3022564, 0.8574660, -0.9855281, 0.8923838
1: -0.3120337, 0.7594209, -0.5588838, 1.0796481, -1.3916818, 1.3183048
2: -0.2382188, 0.8421853, -0.4826456, 1.2362378, -1.4744565, 1.3248309
3: -0.6190236, 0.8373086, -0.9591278, 1.2640674, -1.8830910, 1.7964365
4: -0.4718652, 1.0084939, -0.8620855, 1.4785453, -1.9504105, 1.8705794

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0495071, upper bound: 1.0490162
time: 0.46 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0498067, upper bound: 1.0491330
time: 0.46 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.2275870, 0.7157916, -0.3022564, 0.8574660, -1.0850530, 1.0180480
1: -0.4526966, 0.9158092, -0.5588838, 1.0796481, -1.5323448, 1.4746931
2: -0.3724530, 1.0130844, -0.4826456, 1.2362378, -1.6086907, 1.4957299
3: -0.7981911, 1.0494591, -0.9591278, 1.2640674, -2.0622585, 2.0085869
4: -0.6660109, 1.2261832, -0.8620855, 1.4785453, -2.1445563, 2.0882688

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0520425, upper bound: 1.0509520
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0509838, upper bound: 1.0502371
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0498067, upper bound: 1.0491330
time: 0.46 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1981437, 0.6735660, -0.2350464, 0.7415904, -0.9397342, 0.9086125
1: -0.4075772, 0.8701689, -0.4636297, 0.9372106, -1.3447878, 1.3337986
2: -0.3290880, 0.9651487, -0.3912313, 1.0568323, -1.3859203, 1.3563800
3: -0.7565911, 0.9793513, -0.8212245, 1.0728223, -1.8294134, 1.8005757
4: -0.6072877, 1.1647243, -0.7068326, 1.2774920, -1.8847797, 1.8715570

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 4

Time for candidate selection: 2.71 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0539762
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0539762
time: 0.42 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.2620909, 0.7696464, -0.2350464, 0.7415904, -1.0036813, 1.0046928
1: -0.5045343, 0.9919055, -0.4636297, 0.9372106, -1.4417449, 1.4555352
2: -0.4200620, 1.0960678, -0.3912313, 1.0568323, -1.4768944, 1.4872991
3: -0.8876588, 1.1395187, -0.8212245, 1.0728223, -1.9604812, 1.9607432
4: -0.7470124, 1.3261604, -0.7068326, 1.2774920, -2.0245044, 2.0329931

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 4

Time for candidate selection: 2.63 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0541033
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0541033
time: 0.42 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1981437, 0.6735660, -0.1909873, 0.6899648, -0.8881085, 0.8645533
1: -0.4075772, 0.8701689, -0.4062611, 0.8681383, -1.2757154, 1.2764300
2: -0.3290880, 0.9651487, -0.3319388, 0.9815091, -1.3105972, 1.2970874
3: -0.7565911, 0.9793513, -0.7429183, 0.9865521, -1.7431432, 1.7222695
4: -0.6072877, 1.1647243, -0.6151925, 1.1911318, -1.7984195, 1.7799169

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 4

Time for candidate selection: 2.66 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0534783, upper bound: 1.0539762
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0534783, upper bound: 1.0539762
time: 0.36 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.2620909, 0.7696464, -0.1909873, 0.6899648, -0.9520556, 0.9606337
1: -0.5045343, 0.9919055, -0.4062611, 0.8681383, -1.3726726, 1.3981667
2: -0.4200620, 1.0960678, -0.3319388, 0.9815091, -1.4015712, 1.4280066
3: -0.8876588, 1.1395187, -0.7429183, 0.9865521, -1.8742108, 1.8824370
4: -0.7470124, 1.3261604, -0.6151925, 1.1911318, -1.9381442, 1.9413530

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 4

Time for candidate selection: 2.70 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0534783, upper bound: 1.0541033
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0534783, upper bound: 1.0541033
time: 0.40 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.2013567, 0.6970940, -0.1822524, 0.6750042, -0.8763609, 0.8793464
1: -0.4231896, 0.8673437, -0.3929973, 0.8486877, -1.2718773, 1.2603409
2: -0.3478338, 0.9940906, -0.3201882, 0.9593878, -1.3072217, 1.3142788
3: -0.7603652, 1.0071222, -0.7204593, 0.9646156, -1.7249808, 1.7275815
4: -0.6426137, 1.2188928, -0.5978308, 1.1654682, -1.8080819, 1.8167236

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 4

Time for candidate selection: 2.81 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499520, upper bound: 1.0503022
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0510982, upper bound: 1.0520851
time: 0.46 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.2909008, 0.8438774, -0.1822524, 0.6750042, -0.9659050, 1.0261298
1: -0.5446191, 1.0634959, -0.3929973, 0.8486877, -1.3933069, 1.4564931
2: -0.4665691, 1.2150321, -0.3201882, 0.9593878, -1.4259570, 1.5352204
3: -0.9410769, 1.2407138, -0.7204593, 0.9646156, -1.9056926, 1.9611731
4: -0.8382176, 1.4520541, -0.5978308, 1.1654682, -2.0036860, 2.0498848

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 4

Time for candidate selection: 2.83 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499520, upper bound: 1.0503022
time: 0.48 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0510982, upper bound: 1.0520851
time: 0.47 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.2013567, 0.6970940, -0.2640961, 0.8231158, -1.0244725, 0.9611901
1: -0.4231896, 0.8673437, -0.5030945, 1.0491726, -1.4723623, 1.3704381
2: -0.3478338, 0.9940906, -0.4279521, 1.1928697, -1.5407034, 1.4220426
3: -0.7603652, 1.0071222, -0.8960606, 1.1909752, -1.9513404, 1.9031827
4: -0.6426137, 1.2188928, -0.7792837, 1.4003186, -2.0429323, 1.9981766

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 6

Time for candidate selection: 2.62 seconds

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480409, upper bound: 1.0490002
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0454814, upper bound: 1.0458380
time: 0.50 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.2909008, 0.8438774, -0.2640961, 0.8231158, -1.1140167, 1.1079736
1: -0.5446191, 1.0634959, -0.5030945, 1.0491726, -1.5937917, 1.5665903
2: -0.4665691, 1.2150321, -0.4279521, 1.1928697, -1.6594388, 1.6429842
3: -0.9410769, 1.2407138, -0.8960606, 1.1909752, -2.1320522, 2.1367745
4: -0.8382176, 1.4520541, -0.7792837, 1.4003186, -2.2385364, 2.2313378

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 6

Time for candidate selection: 2.65 seconds

### Candidate
type: B, layer: 3, pos: 17

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0480409, upper bound: 1.0493115
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0434735, upper bound: 1.0441065
time: 0.45 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1981437, 0.6735660, -0.2631326, 0.7698788, -0.9680225, 0.9366986
1: -0.4075772, 0.8701689, -0.5045100, 0.9689885, -1.3765657, 1.3746790
2: -0.3290880, 0.9651487, -0.4292829, 1.0956014, -1.4246894, 1.3944316
3: -0.7565911, 0.9793513, -0.8731403, 1.1296451, -1.8862362, 1.8524916
4: -0.6072877, 1.1647243, -0.7632787, 1.3396218, -1.9469094, 1.9280031

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 4

Time for candidate selection: 2.63 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0535787
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0535787
time: 0.38 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.2620909, 0.7696464, -0.2631326, 0.7698788, -1.0319697, 1.0327790
1: -0.5045343, 0.9919055, -0.5045100, 0.9689885, -1.4735228, 1.4964156
2: -0.4200620, 1.0960678, -0.4292829, 1.0956014, -1.5156634, 1.5253507
3: -0.8876588, 1.1395187, -0.8731403, 1.1296451, -2.0173039, 2.0126591
4: -0.7470124, 1.3261604, -0.7632787, 1.3396218, -2.0866342, 2.0894392

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 4

Time for candidate selection: 2.66 seconds

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0537102
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535777, upper bound: 1.0537102
time: 0.38 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1981437, 0.6735660, -0.2254990, 0.7289720, -0.9271157, 0.8990650
1: -0.4075772, 0.8701689, -0.4573010, 0.9075529, -1.3151300, 1.3274699
2: -0.3290880, 0.9651487, -0.3811812, 1.0387152, -1.3678032, 1.3463299
3: -0.7565911, 0.9793513, -0.8068940, 1.0590353, -1.8156264, 1.7862453
4: -0.6072877, 1.1647243, -0.6912529, 1.2759079, -1.8831956, 1.8559773

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0250000, mid=0.0250000, abs_max=1.1883488893508911
rel_dist={0: [-1.0551075589159629, 1.0551075589159629]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 1149.91 seconds
