## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_1.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 95.41947638718


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396)
1: (-25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411)
2: (-22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495)
3: (-42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475)
4: (-33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989)

## BASE Result
execution time: IAR + LP analysis = 1.71 + 1.63 = 3.35 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -95.4385641, upper bound: 95.4385641


# Binary Search by BASE starts (time budget: 1196.65 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.1666667


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.1666667, mid=0.1666667, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary search (step 1) starts
Candidate diff: 0.0833333


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0833333, mid=0.0833333, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary search (step 2) starts
Candidate diff: 0.0416667


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0416667, mid=0.0416667, abs_max=107.93389892578125
rel_dist={4: [-95.43841502229098, 95.438415022291]}

## Binary search (step 3) starts
Candidate diff: 0.0208333


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0208333, mid=0.0208333, abs_max=107.93389892578125
rel_dist={4: [-95.43801776385524, 95.43801776385524]}

## Binary search (step 4) starts
Candidate diff: 0.0104167


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0104167, mid=0.0104167, abs_max=107.93389892578125
rel_dist={4: [-95.43743645603627, 95.43743645603627]}

## Binary search (step 5) starts
Candidate diff: 0.0052083


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0052083, mid=0.0052083, abs_max=107.93389892578125
rel_dist={4: [-95.43685766005481, 95.43685766005478]}

## Binary search (step 6) starts
Candidate diff: 0.0026042


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0026042, mid=0.0026042, abs_max=107.93389892578125
rel_dist={4: [-95.4365527136626, 95.43655271366259]}

## Binary search (step 7) starts
Candidate diff: 0.0013021


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0013021, mid=0.0013021, abs_max=107.93389892578125
rel_dist={4: [-95.43507596214961, 95.43507596214963]}

## Binary search (step 8) starts
Candidate diff: 0.0006510


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0006510, mid=0.0006510, abs_max=107.93389892578125
rel_dist={4: [-95.4342126817601, 95.4342126817601]}

## Binary search (step 9) starts
Candidate diff: 0.0003255


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0003255, mid=0.0003255, abs_max=107.93389892578125
rel_dist={4: [-95.43372888611276, 95.43372888611276]}

## Binary search (step 10) starts
Candidate diff: 0.0001628


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0001628, mid=0.0001628, abs_max=107.93389892578125
rel_dist={4: [-95.43347375507336, 95.43347375507335]}

## Binary search (step 11) starts
Candidate diff: 0.0000814


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0000814, mid=0.0000814, abs_max=107.93389892578125
rel_dist={4: [-95.43334498914167, 95.43334498914169]}

## Binary search (step 12) starts
Candidate diff: 0.0000407


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0000407, mid=0.0000407, abs_max=107.93389892578125
rel_dist={4: [-95.43328060632305, 95.43328060632305]}

## Binary search (step 13) starts
Candidate diff: 0.0000203


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000203, mid=0.0000203, abs_max=107.93389892578125
rel_dist={4: [-95.43324803845512, 95.43324803845513]}

## Binary search (step 14) starts
Candidate diff: 0.0000102


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000102, mid=0.0000102, abs_max=107.93389892578125
rel_dist={4: [-95.4332314428192, 95.43323144281925]}

## Binary search (step 15) starts
Candidate diff: 0.0000051


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000051, mid=0.0000051, abs_max=107.93389892578125
rel_dist={4: [-95.43322314584034, 95.43322314609429]}

## Binary search (step 16) starts
Candidate diff: 0.0000025


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000025, mid=0.0000025, abs_max=107.93389892578125
rel_dist={4: [-95.43321899120869, 95.43321899969155]}

## Binary search (step 17) starts
Candidate diff: 0.0000013


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000013, mid=0.0000013, abs_max=107.93389892578125
rel_dist={4: [-95.433216924984, 95.43321694589028]}

## Binary search (step 18) starts
Candidate diff: 0.0000006


## IAR start
Binary search (step 18): status=Status.UNKNOWN, low=0.0000000, high=0.0000006, mid=0.0000006, abs_max=107.93389892578125
rel_dist={4: [-95.43321594479565, 95.43321789875597]}

## Binary Search Result
Binary search time: 67.24 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 1129.41 seconds

## Binary search (step 0) starts
Candidate diff: 0.1666667


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4342919, upper bound: 95.4224014
time: 0.56 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.82 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.55 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.55
Output dim: 4, lower bound: -95.4342919, upper bound: 95.4224014
IS_A2, status: Status.VERIFIED, split count: 1, time: 1.55
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -10.8085632, 43.4659424, -16.1209259, 62.1793175, -72.9878693, 59.5868683
1: -17.7322559, 51.5844688, -25.9997044, 73.2714386, -91.0036926, 77.5841675
2: -15.0848141, 54.2866859, -22.6758308, 76.9590302, -92.0438385, 76.9625015
3: -29.5668373, 49.5417938, -42.5341644, 71.9460983, -101.5129242, 92.0759430
4: -22.0560627, 52.4400978, -33.1754951, 74.7584076, -96.8144684, 85.6155930

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.59 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.62 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.99 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 2.99
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 2.99
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
Binary search (step 0): status=Status.VERIFIED, low=0.1666667, high=0.3333333, mid=0.1666667, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4353804, upper bound: 95.4225014
time: 0.57 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.62 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.36 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.36
Output dim: 4, lower bound: -95.4353804, upper bound: 95.4225014
IS_A2, status: Status.VERIFIED, split count: 1, time: 1.36
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -10.8085632, 43.4659424, -16.1209259, 62.1793175, -72.9878693, 59.5868683
1: -17.7322559, 51.5844688, -25.9997044, 73.2714386, -91.0036926, 77.5841675
2: -15.0848141, 54.2866859, -22.6758308, 76.9590302, -92.0438385, 76.9625015
3: -29.5668373, 49.5417938, -42.5341644, 71.9460983, -101.5129242, 92.0759430
4: -22.0560627, 52.4400978, -33.1754951, 74.7584076, -96.8144684, 85.6155930

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.57 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.62 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.28 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 3.28
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 3.28
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
Binary search (step 1): status=Status.VERIFIED, low=0.2500000, high=0.3333333, mid=0.2500000, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary search (step 2) starts
Candidate diff: 0.2916666


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4353804, upper bound: 95.4225014
time: 0.53 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.59 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.28 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.28
Output dim: 4, lower bound: -95.4353804, upper bound: 95.4225014
IS_A2, status: Status.VERIFIED, split count: 1, time: 1.28
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -10.8085632, 43.4659424, -16.1209259, 62.1793175, -72.9878693, 59.5868683
1: -17.7322559, 51.5844688, -25.9997044, 73.2714386, -91.0036926, 77.5841675
2: -15.0848141, 54.2866859, -22.6758308, 76.9590302, -92.0438385, 76.9625015
3: -29.5668373, 49.5417938, -42.5341644, 71.9460983, -101.5129242, 92.0759430
4: -22.0560627, 52.4400978, -33.1754951, 74.7584076, -96.8144684, 85.6155930

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.58 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.54 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.89 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 2.89
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 2.89
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
Binary search (step 2): status=Status.VERIFIED, low=0.2916666, high=0.3333333, mid=0.2916666, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary search (step 3) starts
Candidate diff: 0.3125000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4353804, upper bound: 95.4225014
time: 0.57 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.59 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.32 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.32
Output dim: 4, lower bound: -95.4353804, upper bound: 95.4225014
IS_A2, status: Status.VERIFIED, split count: 1, time: 1.32
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -10.8085632, 43.4659424, -16.1209259, 62.1793175, -72.9878693, 59.5868683
1: -17.7322559, 51.5844688, -25.9997044, 73.2714386, -91.0036926, 77.5841675
2: -15.0848141, 54.2866859, -22.6758308, 76.9590302, -92.0438385, 76.9625015
3: -29.5668373, 49.5417938, -42.5341644, 71.9460983, -101.5129242, 92.0759430
4: -22.0560627, 52.4400978, -33.1754951, 74.7584076, -96.8144684, 85.6155930

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.58 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.54 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.98 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 2.98
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 2.98
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
Binary search (step 3): status=Status.VERIFIED, low=0.3125000, high=0.3333333, mid=0.3125000, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary search (step 4) starts
Candidate diff: 0.3229166


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4353804, upper bound: 95.4225014
time: 0.52 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.58 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.26 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.26
Output dim: 4, lower bound: -95.4353804, upper bound: 95.4225014
IS_A2, status: Status.VERIFIED, split count: 1, time: 1.26
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -10.8085632, 43.4659424, -16.1209259, 62.1793175, -72.9878693, 59.5868683
1: -17.7322559, 51.5844688, -25.9997044, 73.2714386, -91.0036926, 77.5841675
2: -15.0848141, 54.2866859, -22.6758308, 76.9590302, -92.0438385, 76.9625015
3: -29.5668373, 49.5417938, -42.5341644, 71.9460983, -101.5129242, 92.0759430
4: -22.0560627, 52.4400978, -33.1754951, 74.7584076, -96.8144684, 85.6155930

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.60 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.54 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.93 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 2.93
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 2.93
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
Binary search (step 4): status=Status.VERIFIED, low=0.3229166, high=0.3333333, mid=0.3229166, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary search (step 5) starts
Candidate diff: 0.3281250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4353804, upper bound: 95.4225014
time: 0.52 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.54 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.21 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.21
Output dim: 4, lower bound: -95.4353804, upper bound: 95.4225014
IS_A2, status: Status.VERIFIED, split count: 1, time: 1.21
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -10.8085632, 43.4659424, -16.1209259, 62.1793175, -72.9878693, 59.5868683
1: -17.7322559, 51.5844688, -25.9997044, 73.2714386, -91.0036926, 77.5841675
2: -15.0848141, 54.2866859, -22.6758308, 76.9590302, -92.0438385, 76.9625015
3: -29.5668373, 49.5417938, -42.5341644, 71.9460983, -101.5129242, 92.0759430
4: -22.0560627, 52.4400978, -33.1754951, 74.7584076, -96.8144684, 85.6155930

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.56 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.53 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.88 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 2.88
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 2.88
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
Binary search (step 5): status=Status.VERIFIED, low=0.3281250, high=0.3333333, mid=0.3281250, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405608688]}

## Binary search (step 6) starts
Candidate diff: 0.3307291


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4353804, upper bound: 95.4225014
time: 0.52 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.58 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.25 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.25
Output dim: 4, lower bound: -95.4353804, upper bound: 95.4225014
IS_A2, status: Status.VERIFIED, split count: 1, time: 1.25
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -10.8085632, 43.4659424, -16.1209259, 62.1793175, -72.9878693, 59.5868683
1: -17.7322559, 51.5844688, -25.9997044, 73.2714386, -91.0036926, 77.5841675
2: -15.0848141, 54.2866859, -22.6758308, 76.9590302, -92.0438385, 76.9625015
3: -29.5668373, 49.5417938, -42.5341644, 71.9460983, -101.5129242, 92.0759430
4: -22.0560627, 52.4400978, -33.1754951, 74.7584076, -96.8144684, 85.6155930

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.58 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.56 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.91 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 2.91
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 2.91
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
Binary search (step 6): status=Status.VERIFIED, low=0.3307291, high=0.3333333, mid=0.3307291, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary search (step 7) starts
Candidate diff: 0.3320312


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4353804, upper bound: 95.4225014
time: 0.52 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.59 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.26 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.26
Output dim: 4, lower bound: -95.4353804, upper bound: 95.4225014
IS_A2, status: Status.VERIFIED, split count: 1, time: 1.26
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -10.8085632, 43.4659424, -16.1209259, 62.1793175, -72.9878693, 59.5868683
1: -17.7322559, 51.5844688, -25.9997044, 73.2714386, -91.0036926, 77.5841675
2: -15.0848141, 54.2866859, -22.6758308, 76.9590302, -92.0438385, 76.9625015
3: -29.5668373, 49.5417938, -42.5341644, 71.9460983, -101.5129242, 92.0759430
4: -22.0560627, 52.4400978, -33.1754951, 74.7584076, -96.8144684, 85.6155930

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.55 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.54 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.87 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 2.87
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 2.87
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
Binary search (step 7): status=Status.VERIFIED, low=0.3320312, high=0.3333333, mid=0.3320312, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary search (step 8) starts
Candidate diff: 0.3326823


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4353804, upper bound: 95.4225014
time: 0.58 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.58 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.30 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.30
Output dim: 4, lower bound: -95.4353804, upper bound: 95.4225014
IS_A2, status: Status.VERIFIED, split count: 1, time: 1.30
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -10.8085632, 43.4659424, -16.1209259, 62.1793175, -72.9878693, 59.5868683
1: -17.7322559, 51.5844688, -25.9997044, 73.2714386, -91.0036926, 77.5841675
2: -15.0848141, 54.2866859, -22.6758308, 76.9590302, -92.0438385, 76.9625015
3: -29.5668373, 49.5417938, -42.5341644, 71.9460983, -101.5129242, 92.0759430
4: -22.0560627, 52.4400978, -33.1754951, 74.7584076, -96.8144684, 85.6155930

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.60 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.55 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.95 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 2.95
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 2.95
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
Binary search (step 8): status=Status.VERIFIED, low=0.3326823, high=0.3333333, mid=0.3326823, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary search (step 9) starts
Candidate diff: 0.3330078


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4353804, upper bound: 95.4225014
time: 0.52 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.59 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.27 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.27
Output dim: 4, lower bound: -95.4353804, upper bound: 95.4225014
IS_A2, status: Status.VERIFIED, split count: 1, time: 1.27
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -10.8085632, 43.4659424, -16.1209259, 62.1793175, -72.9878693, 59.5868683
1: -17.7322559, 51.5844688, -25.9997044, 73.2714386, -91.0036926, 77.5841675
2: -15.0848141, 54.2866859, -22.6758308, 76.9590302, -92.0438385, 76.9625015
3: -29.5668373, 49.5417938, -42.5341644, 71.9460983, -101.5129242, 92.0759430
4: -22.0560627, 52.4400978, -33.1754951, 74.7584076, -96.8144684, 85.6155930

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.57 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.54 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.92 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 2.92
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 2.92
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
Binary search (step 9): status=Status.VERIFIED, low=0.3330078, high=0.3333333, mid=0.3330078, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary search (step 10) starts
Candidate diff: 0.3331706


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4353804, upper bound: 95.4225014
time: 0.52 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.59 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.26 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.26
Output dim: 4, lower bound: -95.4353804, upper bound: 95.4225014
IS_A2, status: Status.VERIFIED, split count: 1, time: 1.26
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -10.8085632, 43.4659424, -16.1209259, 62.1793175, -72.9878693, 59.5868683
1: -17.7322559, 51.5844688, -25.9997044, 73.2714386, -91.0036926, 77.5841675
2: -15.0848141, 54.2866859, -22.6758308, 76.9590302, -92.0438385, 76.9625015
3: -29.5668373, 49.5417938, -42.5341644, 71.9460983, -101.5129242, 92.0759430
4: -22.0560627, 52.4400978, -33.1754951, 74.7584076, -96.8144684, 85.6155930

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.57 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.56 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.91 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 2.91
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 2.91
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
Binary search (step 10): status=Status.VERIFIED, low=0.3331706, high=0.3333333, mid=0.3331706, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary search (step 11) starts
Candidate diff: 0.3332519


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4353804, upper bound: 95.4225014
time: 0.51 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.61 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.26 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.26
Output dim: 4, lower bound: -95.4353804, upper bound: 95.4225014
IS_A2, status: Status.VERIFIED, split count: 1, time: 1.26
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -10.8085632, 43.4659424, -16.1209259, 62.1793175, -72.9878693, 59.5868683
1: -17.7322559, 51.5844688, -25.9997044, 73.2714386, -91.0036926, 77.5841675
2: -15.0848141, 54.2866859, -22.6758308, 76.9590302, -92.0438385, 76.9625015
3: -29.5668373, 49.5417938, -42.5341644, 71.9460983, -101.5129242, 92.0759430
4: -22.0560627, 52.4400978, -33.1754951, 74.7584076, -96.8144684, 85.6155930

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.57 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.55 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.97 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 2.97
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 2.97
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
Binary search (step 11): status=Status.VERIFIED, low=0.3332519, high=0.3333333, mid=0.3332519, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary search (step 12) starts
Candidate diff: 0.3332926


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4353804, upper bound: 95.4225014
time: 0.51 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.58 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.24 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.24
Output dim: 4, lower bound: -95.4353804, upper bound: 95.4225014
IS_A2, status: Status.VERIFIED, split count: 1, time: 1.24
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -10.8085632, 43.4659424, -16.1209259, 62.1793175, -72.9878693, 59.5868683
1: -17.7322559, 51.5844688, -25.9997044, 73.2714386, -91.0036926, 77.5841675
2: -15.0848141, 54.2866859, -22.6758308, 76.9590302, -92.0438385, 76.9625015
3: -29.5668373, 49.5417938, -42.5341644, 71.9460983, -101.5129242, 92.0759430
4: -22.0560627, 52.4400978, -33.1754951, 74.7584076, -96.8144684, 85.6155930

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.57 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.55 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.92 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 2.92
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 2.92
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
Binary search (step 12): status=Status.VERIFIED, low=0.3332926, high=0.3333333, mid=0.3332926, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary search (step 13) starts
Candidate diff: 0.3333130


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4353804, upper bound: 95.4225014
time: 0.52 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.58 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.26 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.26
Output dim: 4, lower bound: -95.4353804, upper bound: 95.4225014
IS_A2, status: Status.VERIFIED, split count: 1, time: 1.26
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -10.8085632, 43.4659424, -16.1209259, 62.1793175, -72.9878693, 59.5868683
1: -17.7322559, 51.5844688, -25.9997044, 73.2714386, -91.0036926, 77.5841675
2: -15.0848141, 54.2866859, -22.6758308, 76.9590302, -92.0438385, 76.9625015
3: -29.5668373, 49.5417938, -42.5341644, 71.9460983, -101.5129242, 92.0759430
4: -22.0560627, 52.4400978, -33.1754951, 74.7584076, -96.8144684, 85.6155930

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.56 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.55 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.90 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 2.90
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 2.90
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
Binary search (step 13): status=Status.VERIFIED, low=0.3333130, high=0.3333333, mid=0.3333130, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary search (step 14) starts
Candidate diff: 0.3333231


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4353804, upper bound: 95.4225014
time: 0.52 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.58 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.25 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.25
Output dim: 4, lower bound: -95.4353804, upper bound: 95.4225014
IS_A2, status: Status.VERIFIED, split count: 1, time: 1.25
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -10.8085632, 43.4659424, -16.1209259, 62.1793175, -72.9878693, 59.5868683
1: -17.7322559, 51.5844688, -25.9997044, 73.2714386, -91.0036926, 77.5841675
2: -15.0848141, 54.2866859, -22.6758308, 76.9590302, -92.0438385, 76.9625015
3: -29.5668373, 49.5417938, -42.5341644, 71.9460983, -101.5129242, 92.0759430
4: -22.0560627, 52.4400978, -33.1754951, 74.7584076, -96.8144684, 85.6155930

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.56 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.56 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.87 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 2.87
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 2.87
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
Binary search (step 14): status=Status.VERIFIED, low=0.3333231, high=0.3333333, mid=0.3333231, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary search (step 15) starts
Candidate diff: 0.3333282


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4353804, upper bound: 95.4225014
time: 0.51 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.59 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.25 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.25
Output dim: 4, lower bound: -95.4353804, upper bound: 95.4225014
IS_A2, status: Status.VERIFIED, split count: 1, time: 1.25
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -10.8085632, 43.4659424, -16.1209259, 62.1793175, -72.9878693, 59.5868683
1: -17.7322559, 51.5844688, -25.9997044, 73.2714386, -91.0036926, 77.5841675
2: -15.0848141, 54.2866859, -22.6758308, 76.9590302, -92.0438385, 76.9625015
3: -29.5668373, 49.5417938, -42.5341644, 71.9460983, -101.5129242, 92.0759430
4: -22.0560627, 52.4400978, -33.1754951, 74.7584076, -96.8144684, 85.6155930

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.57 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.60 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.95 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 2.95
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 2.95
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
Binary search (step 15): status=Status.VERIFIED, low=0.3333282, high=0.3333333, mid=0.3333282, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary search (step 16) starts
Candidate diff: 0.3333308


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4353804, upper bound: 95.4225014
time: 0.51 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.58 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.25 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.25
Output dim: 4, lower bound: -95.4353804, upper bound: 95.4225014
IS_A2, status: Status.VERIFIED, split count: 1, time: 1.25
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -10.8085632, 43.4659424, -16.1209259, 62.1793175, -72.9878693, 59.5868683
1: -17.7322559, 51.5844688, -25.9997044, 73.2714386, -91.0036926, 77.5841675
2: -15.0848141, 54.2866859, -22.6758308, 76.9590302, -92.0438385, 76.9625015
3: -29.5668373, 49.5417938, -42.5341644, 71.9460983, -101.5129242, 92.0759430
4: -22.0560627, 52.4400978, -33.1754951, 74.7584076, -96.8144684, 85.6155930

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.56 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.55 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.93 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 2.93
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 2.93
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
Binary search (step 16): status=Status.VERIFIED, low=0.3333308, high=0.3333333, mid=0.3333308, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary search (step 17) starts
Candidate diff: 0.3333320


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4353804, upper bound: 95.4225014
time: 0.51 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.53 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.20 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.20
Output dim: 4, lower bound: -95.4353804, upper bound: 95.4225014
IS_A2, status: Status.VERIFIED, split count: 1, time: 1.20
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -10.8085632, 43.4659424, -16.1209259, 62.1793175, -72.9878693, 59.5868683
1: -17.7322559, 51.5844688, -25.9997044, 73.2714386, -91.0036926, 77.5841675
2: -15.0848141, 54.2866859, -22.6758308, 76.9590302, -92.0438385, 76.9625015
3: -29.5668373, 49.5417938, -42.5341644, 71.9460983, -101.5129242, 92.0759430
4: -22.0560627, 52.4400978, -33.1754951, 74.7584076, -96.8144684, 85.6155930

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.55 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.55 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.92 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 2.92
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 2.92
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
Binary search (step 17): status=Status.VERIFIED, low=0.3333320, high=0.3333333, mid=0.3333320, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary search (step 18) starts
Candidate diff: 0.3333327


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4353804, upper bound: 95.4225014
time: 0.51 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.58 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.24 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.24
Output dim: 4, lower bound: -95.4353804, upper bound: 95.4225014
IS_A2, status: Status.VERIFIED, split count: 1, time: 1.24
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -10.8085632, 43.4659424, -16.1209259, 62.1793175, -72.9878693, 59.5868683
1: -17.7322559, 51.5844688, -25.9997044, 73.2714386, -91.0036926, 77.5841675
2: -15.0848141, 54.2866859, -22.6758308, 76.9590302, -92.0438385, 76.9625015
3: -29.5668373, 49.5417938, -42.5341644, 71.9460983, -101.5129242, 92.0759430
4: -22.0560627, 52.4400978, -33.1754951, 74.7584076, -96.8144684, 85.6155930

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.56 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.55 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.92 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 2.92
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 2.92
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
Binary search (step 18): status=Status.VERIFIED, low=0.3333327, high=0.3333333, mid=0.3333327, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.3333326776822787
execution time: 145.05 seconds
