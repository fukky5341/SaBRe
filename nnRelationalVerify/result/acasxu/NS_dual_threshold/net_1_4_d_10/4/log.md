## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 4)
Time budget: 420 seconds
Split limit: 100
Threshold: 96.5219627187


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-38.2100830, 62.7565804, -38.2100830, 62.7565804, -100.9666443, 100.9666443)
1: (-41.6995087, 54.3409386, -41.6995087, 54.3409386, -96.0404510, 96.0404358)
2: (-42.7050247, 54.2436867, -42.7050247, 54.2436867, -96.9486923, 96.9486923)
3: (-48.9462700, 63.1773834, -48.9462700, 63.1773834, -112.1236496, 112.1236496)
4: (-45.2550850, 62.9522591, -45.2550850, 62.9522591, -108.2073441, 108.2073441)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.84 + 1.86 = 2.70 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -96.6185813, upper bound: 96.6185813

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6073850, upper bound: 96.6144003
time: 0.75 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6146532, upper bound: 96.6146532
time: 0.90 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.75 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.75
Output dim: 4, lower bound: -96.6073850, upper bound: 96.6144003
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.75
Output dim: 4, lower bound: -96.6146532, upper bound: 96.6146532

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -37.5410461, 62.1251144, -38.2100830, 62.7565804, -100.2976227, 100.3351898
1: -41.0108490, 53.9394341, -41.6995087, 54.3409386, -95.3517914, 95.6389465
2: -41.9839706, 53.8069801, -42.7050247, 54.2436867, -96.2276611, 96.5120010
3: -48.2026939, 62.6904144, -48.9462700, 63.1773834, -111.3800812, 111.6366806
4: -44.5580063, 62.4137344, -45.2550850, 62.9522591, -107.5102692, 107.6688232

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6065214, upper bound: 96.6044178
time: 0.63 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6065214, upper bound: 96.6141926
time: 0.84 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -36.6395493, 60.5369682, -38.2100830, 62.7565804, -99.3961334, 98.7470398
1: -40.0196915, 52.3880043, -41.6995087, 54.3409386, -94.3606262, 94.0874939
2: -40.9790764, 52.2644157, -42.7050247, 54.2436867, -95.2227478, 94.9694366
3: -47.0237045, 60.9028969, -48.9462700, 63.1773834, -110.2010803, 109.8491669
4: -43.5041313, 60.6233826, -45.2550850, 62.9522591, -106.4563904, 105.8784637

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5310459, upper bound: 96.5952749
time: 0.79 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6113031, upper bound: 96.6113032
time: 0.60 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.29 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.29
Output dim: 4, lower bound: -96.6065214, upper bound: 96.6044178
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.29
Output dim: 4, lower bound: -96.6065214, upper bound: 96.6141926
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 2.29
Output dim: 4, lower bound: -96.5310459, upper bound: 96.5952749
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 2.29
Output dim: 4, lower bound: -96.6113031, upper bound: 96.6113032

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -37.5410461, 62.1251144, -37.5779495, 62.1574745, -99.6985168, 99.7030640
1: -41.0108490, 53.9394341, -41.0252304, 53.7039070, -94.7147522, 94.9646606
2: -41.9839706, 53.8069801, -42.0214386, 53.5866394, -95.5706100, 95.8283920
3: -48.2026939, 62.6904144, -48.2054901, 62.4242668, -110.6269455, 110.8959045
4: -44.5580063, 62.4137344, -44.5850372, 62.1696434, -106.7276459, 106.9987717

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5287166, upper bound: 96.5862474
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5935643, upper bound: 96.6017186
time: 0.89 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -37.5410461, 62.1251144, -38.1697311, 62.7158737, -100.2569199, 100.2948456
1: -41.0108490, 53.9394341, -41.6565094, 54.3016815, -95.3125305, 95.5959473
2: -41.9839706, 53.8069801, -42.6617813, 54.2028618, -96.1868286, 96.4687653
3: -48.2026939, 62.6904144, -48.8994026, 63.1311684, -111.3338623, 111.5898132
4: -44.5580063, 62.4137344, -45.2133522, 62.9031982, -107.4612045, 107.6270905

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5287166, upper bound: 96.5862474
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5942393, upper bound: 96.6091845
time: 0.82 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -22.0901928, 42.5159149, -36.3742218, 60.3238182, -82.4140015, 78.8901367
1: -24.2454433, 35.0445175, -39.6944351, 51.9614906, -76.2069092, 74.7389526
2: -24.8877068, 34.7209930, -40.6514015, 51.8626137, -76.7503204, 75.3723907
3: -29.0086060, 40.5432358, -46.6203918, 60.4010353, -89.4096146, 87.1636200
4: -27.8547173, 39.6899300, -43.1367874, 60.1164207, -87.9711380, 82.8267212

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5301221, upper bound: 96.5872825
time: 0.69 seconds

## Relational analysis of NS_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5308423, upper bound: 96.5950364
time: 0.91 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -45.8915329, 77.3059616, -37.5084381, 61.7592354, -107.6507721, 114.8143692
1: -50.1652069, 65.5790253, -40.9360428, 53.4188156, -103.5840225, 106.5150681
2: -51.3951492, 65.7022934, -41.9247131, 53.3292389, -104.7243881, 107.6270065
3: -59.1215591, 76.0324402, -48.0760803, 62.0458717, -121.1674194, 124.1085205
4: -54.1838837, 76.2210464, -44.4276314, 61.8733482, -116.0572357, 120.6486511

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5952749, upper bound: 96.5310459
time: 0.91 seconds

## Relational analysis of NS_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5952749, upper bound: 96.5310459
time: 1.06 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.77 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.77
Output dim: 4, lower bound: -96.5287166, upper bound: 96.5862474
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.77
Output dim: 4, lower bound: -96.5935643, upper bound: 96.6017186
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.77
Output dim: 4, lower bound: -96.5287166, upper bound: 96.5862474
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.77
Output dim: 4, lower bound: -96.5942393, upper bound: 96.6091845
NS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 2.77
Output dim: 4, lower bound: -96.5301221, upper bound: 96.5872825
NS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 2.77
Output dim: 4, lower bound: -96.5308423, upper bound: 96.5950364
NS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 2.77
Output dim: 4, lower bound: -96.5952749, upper bound: 96.5310459
NS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 2.77
Output dim: 4, lower bound: -96.5952749, upper bound: 96.5310459

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -22.5888329, 43.2161484, -35.7402687, 59.7223396, -82.3111496, 78.9564209
1: -24.7766838, 35.7365608, -39.0180626, 51.3241272, -76.1008072, 74.7546234
2: -25.4296799, 35.4515381, -39.9663811, 51.2037125, -76.6333847, 75.4179077
3: -29.6232643, 41.3063087, -45.8772964, 59.6481400, -89.2713928, 87.1836014
4: -28.3481541, 40.5114021, -42.4687500, 59.3321953, -87.6803513, 82.9801483

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5276922, upper bound: 96.5774402
time: 0.99 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5276922, upper bound: 96.5862474
time: 0.94 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -46.8327942, 78.9024353, -36.8759956, 61.1586418, -107.9914398, 115.7784119
1: -51.1982498, 67.0478745, -40.2613182, 52.7805557, -103.9787903, 107.3091736
2: -52.4324837, 67.1990433, -41.2409172, 52.6711349, -105.1036224, 108.4399414
3: -60.3272438, 77.6979752, -47.3344688, 61.2923660, -121.6195984, 125.0324326
4: -55.2086487, 77.9360428, -43.7568474, 61.0890656, -116.2977142, 121.6928864

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5807428, upper bound: 96.5499522
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5910003, upper bound: 96.6008679
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -22.5888329, 43.2161484, -36.3337708, 60.2829590, -82.8717880, 79.5499191
1: -24.7766838, 35.7365608, -39.6513062, 51.9222260, -76.6989136, 75.3878632
2: -25.4296799, 35.4515381, -40.6081085, 51.8216820, -77.2513580, 76.0596466
3: -29.6232643, 41.3063087, -46.5734367, 60.3548393, -89.9780960, 87.8797455
4: -28.3481541, 40.5114021, -43.0951271, 60.0672073, -88.4153595, 83.6065292

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5165007, upper bound: 96.5178301
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5165007, upper bound: 96.5940222
time: 0.89 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -46.8327942, 78.9024353, -37.4680252, 61.7185898, -108.5513840, 116.3704376
1: -51.1982498, 67.0478745, -40.8929901, 53.3796310, -104.5778503, 107.9408646
2: -52.4324837, 67.1990433, -41.8814011, 53.2884598, -105.7209473, 109.0804443
3: -60.3272438, 77.6979752, -48.0290794, 61.9998550, -122.3271027, 125.7270508
4: -55.2086487, 77.9360428, -44.3858032, 61.8242416, -117.0328903, 122.3218460

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5809792, upper bound: 96.5291290
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5809792, upper bound: 96.6091846
time: 0.77 seconds

## BFS NS instance: NS_A2_A1_B1

### Backsubstitution after applying NS history:
0: -22.0901928, 42.5159149, -35.7402687, 59.7223396, -81.8125000, 78.2561798
1: -24.2454433, 35.0445175, -39.0180626, 51.3241272, -75.5695648, 74.0625763
2: -24.8877068, 34.7209930, -39.9663811, 51.2037125, -76.0914154, 74.6873703
3: -29.0086060, 40.5432358, -45.8772964, 59.6481400, -88.6567230, 86.4205170
4: -27.8547173, 39.6899300, -42.4687500, 59.3321953, -87.1869125, 82.1586761

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_A1_B1_A1

### Relational analysis result of NS_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5210057, upper bound: 96.5694693
time: 1.28 seconds

## Relational analysis of NS_A2_A1_B1_A2

### Relational analysis result of NS_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5293982, upper bound: 96.5775801
time: 0.97 seconds

## BFS NS instance: NS_A2_A1_B2

### Backsubstitution after applying NS history:
0: -22.0901928, 42.5159149, -36.3337708, 60.2829590, -82.3731384, 78.8496857
1: -24.2454433, 35.0445175, -39.6513062, 51.9222260, -76.1676712, 74.6958237
2: -24.8877068, 34.7209930, -40.6081085, 51.8216820, -76.7093887, 75.3291016
3: -29.0086060, 40.5432358, -46.5734367, 60.3548393, -89.3634262, 87.1166687
4: -27.8547173, 39.6899300, -43.0951271, 60.0672073, -87.9219208, 82.7850571

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_A1_B2_B1

### Relational analysis result of NS_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5177675, upper bound: 96.5177664
time: 1.03 seconds

## Relational analysis of NS_A2_A1_B2_B2

### Relational analysis result of NS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5177675, upper bound: 96.5950364
time: 0.77 seconds

## BFS NS instance: NS_A2_A2_B1

### Backsubstitution after applying NS history:
0: -45.8915329, 77.3059616, -23.6491947, 44.5811996, -90.4727325, 100.9551544
1: -50.1652069, 65.5790253, -25.9020424, 36.9293671, -87.0945740, 91.4810562
2: -51.3951492, 65.7022934, -26.6062851, 36.6187668, -88.0138931, 92.3085709
3: -59.1215591, 76.0324402, -30.8861237, 42.7392807, -101.8608398, 106.9185638
4: -54.1838837, 76.2210464, -29.5871983, 41.9198265, -96.1037140, 105.8082428

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_A2_B1_A1

### Relational analysis result of NS_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5872823, upper bound: 96.5301220
time: 0.85 seconds

## Relational analysis of NS_A2_A2_B1_A2

### Relational analysis result of NS_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5950361, upper bound: 96.5308423
time: 1.05 seconds

## BFS NS instance: NS_A2_A2_B2

### Backsubstitution after applying NS history:
0: -45.8915329, 77.3059616, -47.7588348, 79.9911652, -125.8826981, 125.0647964
1: -50.1652069, 65.5790253, -52.1639366, 67.9121017, -118.0773087, 117.7429581
2: -51.3951492, 65.7022934, -53.4510193, 68.0761642, -119.4713135, 119.1533127
3: -59.1215591, 76.0324402, -61.4133224, 78.7076035, -137.8291626, 137.4457245
4: -54.1838837, 76.2210464, -56.2418251, 79.0030441, -133.1869202, 132.4628448

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_A2_B2_B1

### Relational analysis result of NS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5942707, upper bound: 96.5944377
time: 1.17 seconds

## Relational analysis of NS_A2_A2_B2_B2

### Relational analysis result of NS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5942708, upper bound: 96.6107614
time: 0.94 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.17 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 4, lower bound: -96.5276922, upper bound: 96.5774402
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 4, lower bound: -96.5276922, upper bound: 96.5862474
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 4, lower bound: -96.5807428, upper bound: 96.5499522
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 4, lower bound: -96.5910003, upper bound: 96.6008679
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.17
Output dim: 4, lower bound: -96.5165007, upper bound: 96.5178301
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 4, lower bound: -96.5165007, upper bound: 96.5940222
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 4, lower bound: -96.5809792, upper bound: 96.5291290
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 4, lower bound: -96.5809792, upper bound: 96.6091846
NS_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 4, lower bound: -96.5210057, upper bound: 96.5694693
NS_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 4, lower bound: -96.5293982, upper bound: 96.5775801
NS_A2_A1_B2_B1, status: Status.VERIFIED, split count: 4, time: 3.17
Output dim: 4, lower bound: -96.5177675, upper bound: 96.5177664
NS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 4, lower bound: -96.5177675, upper bound: 96.5950364
NS_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 4, lower bound: -96.5872823, upper bound: 96.5301220
NS_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 4, lower bound: -96.5950361, upper bound: 96.5308423
NS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 4, lower bound: -96.5942707, upper bound: 96.5944377
NS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 4, lower bound: -96.5942708, upper bound: 96.6107614

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -22.5888329, 43.2161484, -35.1516113, 59.2091713, -81.7980042, 78.3677597
1: -24.7766838, 35.7365608, -38.4155922, 51.0388718, -75.8155518, 74.1521454
2: -25.4296799, 35.4515381, -39.3333931, 50.8798904, -76.3095703, 74.7849274
3: -29.6232643, 41.3063087, -45.2213821, 59.2580223, -88.8812866, 86.5276947
4: -28.3481541, 40.5114021, -41.8584366, 58.9250221, -87.2731781, 82.3698349

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5203875, upper bound: 96.5750033
time: 0.91 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5276922, upper bound: 96.5774402
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -22.5888329, 43.2161484, -34.1786346, 57.5382462, -80.1270752, 77.3947830
1: -24.7766838, 35.7365608, -37.3464203, 49.3948364, -74.1715240, 73.0829773
2: -25.4296799, 35.4515381, -38.2511826, 49.2476044, -74.6772766, 73.7027130
3: -29.6232643, 41.3063087, -43.9584045, 57.4022827, -87.0255356, 85.2647095
4: -28.3481541, 40.5114021, -40.7358665, 57.0290222, -85.3771744, 81.2472687

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5203875, upper bound: 96.5787537
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5276922, upper bound: 96.5862474
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -46.8327942, 78.9024353, -34.3436432, 58.0345650, -104.8673553, 113.2460632
1: -51.1982498, 67.0478745, -37.5347443, 49.8279419, -101.0261841, 104.5826035
2: -52.4324837, 67.1990433, -38.4794235, 49.7207184, -102.1531982, 105.6784592
3: -60.3272438, 77.6979752, -44.2864571, 57.8407898, -118.1680298, 121.9844360
4: -55.2086487, 77.9360428, -40.9803696, 57.5617218, -112.7703705, 118.9164124

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5783359, upper bound: 96.5498290
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5783359, upper bound: 96.5499522
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -46.8327942, 78.9024353, -35.9303055, 60.5144310, -107.3472290, 114.8327255
1: -51.1982498, 67.0478745, -39.2484093, 51.9334297, -103.1316605, 106.2962646
2: -52.4324837, 67.1990433, -40.2091751, 51.8252335, -104.2577209, 107.4082108
3: -60.3272438, 77.6979752, -46.3102989, 60.2947693, -120.6220016, 124.0082474
4: -55.2086487, 77.9360428, -42.7250977, 60.0578003, -115.2664490, 120.6611404

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5885935, upper bound: 96.6007448
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5885935, upper bound: 96.6007448
time: 1.09 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -22.5888329, 43.2161484, -47.7190552, 79.9503479, -102.5391769, 90.9351959
1: -24.7766838, 35.7365608, -52.1214371, 67.8727264, -92.6493988, 87.8580017
2: -25.4296799, 35.4515381, -53.4082642, 68.0354233, -93.4651031, 88.8598022
3: -29.6232643, 41.3063087, -61.3669090, 78.6607513, -108.2840118, 102.6732178
4: -28.3481541, 40.5114021, -56.2001648, 78.9533081, -107.3014603, 96.7115555

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5125749, upper bound: 96.5924925
time: 0.96 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B2_B1

### Relational analysis result of NS_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5165007, upper bound: 96.5810424
time: 1.09 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2

### Relational analysis result of NS_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5165007, upper bound: 96.5940222
time: 1.00 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -46.8327942, 78.9024353, -23.6173382, 44.5552788, -91.3880768, 102.5197525
1: -51.1982498, 67.0478745, -25.8699112, 36.9046631, -88.1028900, 92.9177704
2: -52.4324837, 67.1990433, -26.5733566, 36.5929985, -89.0254822, 93.7723923
3: -60.3272438, 77.6979752, -30.8537045, 42.7098389, -103.0370789, 108.5516815
4: -55.2086487, 77.9360428, -29.5615196, 41.8891945, -97.0978394, 107.4975586

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B1_B1

### Relational analysis result of NS_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5809789, upper bound: 96.5278624
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A2_B1_B2

### Relational analysis result of NS_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5809789, upper bound: 96.5291289
time: 0.97 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -46.8327942, 78.9024353, -47.7190552, 79.9503479, -126.7831421, 126.6214676
1: -51.1982498, 67.0478745, -52.1214371, 67.8727264, -119.0709610, 119.1692963
2: -52.4324837, 67.1990433, -53.4082642, 68.0354233, -120.4679108, 120.6073074
3: -60.3272438, 77.6979752, -61.3669090, 78.6607513, -138.9879913, 139.0648804
4: -55.2086487, 77.9360428, -56.2001648, 78.9533081, -134.1619415, 134.1362000

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B2_B1

### Relational analysis result of NS_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5809792, upper bound: 96.5923412
time: 0.79 seconds

## Relational analysis of NS_A1_B2_A2_B2_B2

### Relational analysis result of NS_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5809792, upper bound: 96.6027441
time: 0.78 seconds

## BFS NS instance: NS_A2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -16.0013294, 33.7018280, -34.4706841, 57.9669991, -73.9683304, 68.1725159
1: -17.6864433, 27.2826900, -37.6493912, 49.7624969, -67.4489365, 64.9320831
2: -18.1214790, 26.9609604, -38.5716019, 49.6192322, -67.7407074, 65.5325623
3: -21.4284306, 31.4723530, -44.3082542, 57.8246040, -79.2530212, 75.7806091
4: -21.0283413, 30.6083851, -41.0656319, 57.4749374, -78.5032730, 71.6740036

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_A1_B1_A1_A1

### Relational analysis result of NS_A2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5210024, upper bound: 96.5672577
time: 1.00 seconds

## Relational analysis of NS_A2_A1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_A1_B1_A1_B1

### Relational analysis result of NS_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5209323, upper bound: 96.5694693
time: 0.67 seconds

## Relational analysis of NS_A2_A1_B1_A1_B2

### Relational analysis result of NS_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5175975, upper bound: 96.5573705
time: 1.11 seconds

## BFS NS instance: NS_A2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -20.6750565, 40.6228981, -35.7167320, 59.6906242, -80.3656769, 76.3396301
1: -22.7359962, 33.3683281, -38.9927673, 51.2954636, -74.0314484, 72.3610992
2: -23.3511162, 33.0451317, -39.9405899, 51.1750793, -74.5261993, 72.9857178
3: -27.3136444, 38.5819588, -45.8487663, 59.6146278, -86.9282684, 84.4307175
4: -26.3192291, 37.7103271, -42.4417076, 59.2983360, -85.6175613, 80.1520233

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_A1_B1_A2_B1

### Relational analysis result of NS_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5273141, upper bound: 96.5736455
time: 1.32 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2

### Relational analysis result of NS_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5273141, upper bound: 96.5775801
time: 1.31 seconds

## BFS NS instance: NS_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -22.0901928, 42.5159149, -47.7190552, 79.9503479, -102.0405273, 90.2349548
1: -24.2454433, 35.0445175, -52.1214371, 67.8727264, -92.1181564, 87.1659546
2: -24.8877068, 34.7209930, -53.4082642, 68.0354233, -92.9231262, 88.1292572
3: -29.0086060, 40.5432358, -61.3669090, 78.6607513, -107.6693420, 101.9101334
4: -27.8547173, 39.6899300, -56.2001648, 78.9533081, -106.8080292, 95.8900909

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_A1_B2_B2_A1

### Relational analysis result of NS_A2_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5136112, upper bound: 96.5121750
time: 0.71 seconds

## Relational analysis of NS_A2_A1_B2_B2_A2

### Relational analysis result of NS_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5097627, upper bound: 96.5934604
time: 0.77 seconds

## BFS NS instance: NS_A2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -45.2581558, 76.7053070, -23.6491947, 44.5811996, -89.8393555, 100.3544922
1: -49.4900398, 64.9334412, -25.9020424, 36.9293671, -86.4194031, 90.8354797
2: -50.7084846, 65.0377502, -26.6062851, 36.6187668, -87.3272324, 91.6440353
3: -58.3772240, 75.2664337, -30.8861237, 42.7392807, -101.1165009, 106.1525574
4: -53.5063324, 75.4218216, -29.5871983, 41.9198265, -95.4261627, 105.0090179

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_A2_B1_A1_B1

### Relational analysis result of NS_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5702302, upper bound: 96.5006154
time: 0.81 seconds

## Relational analysis of NS_A2_A2_B1_A1_B2

### Relational analysis result of NS_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5862742, upper bound: 96.5301141
time: 1.27 seconds

## BFS NS instance: NS_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -45.8524361, 77.2659302, -23.6491947, 44.5811996, -90.4336243, 100.9151230
1: -50.1235352, 65.5404968, -25.9020424, 36.9293671, -87.0528946, 91.4425278
2: -51.3530960, 65.6622772, -26.6062851, 36.6187668, -87.9718628, 92.2685623
3: -59.0760727, 75.9864731, -30.8861237, 42.7392807, -101.8153534, 106.8725967
4: -54.1428375, 76.1720734, -29.5871983, 41.9198265, -96.0626678, 105.7592697

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_A2_B1_A2_B1

### Relational analysis result of NS_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5853497, upper bound: 96.5031326
time: 0.80 seconds

## Relational analysis of NS_A2_A2_B1_A2_B2

### Relational analysis result of NS_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5940258, upper bound: 96.5308331
time: 0.84 seconds

## BFS NS instance: NS_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -45.8915329, 77.3059616, -46.8341064, 78.9050522, -124.7965851, 124.1400681
1: -50.1652069, 65.5790253, -51.1997833, 67.0504303, -117.2156372, 116.7788010
2: -51.3951492, 65.7022934, -52.4339867, 67.2015381, -118.5966873, 118.1362762
3: -59.1215591, 76.0324402, -60.3291512, 77.7008514, -136.8224030, 136.3615875
4: -54.1838837, 76.2210464, -55.2105370, 77.9386902, -132.1225586, 131.4315338

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_A2_B2_B1_A1

### Relational analysis result of NS_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5977133, upper bound: 96.5862629
time: 0.87 seconds

## Relational analysis of NS_A2_A2_B2_B1_A2

### Relational analysis result of NS_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5976149, upper bound: 96.5869317
time: 1.06 seconds

## BFS NS instance: NS_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -45.8915329, 77.3059616, -46.0706635, 77.7413635, -123.6328964, 123.3766251
1: -50.1652069, 65.5790253, -50.3664093, 65.9208984, -116.0861053, 115.9454193
2: -51.3951492, 65.7022934, -51.6066208, 66.0480347, -117.4431839, 117.3089142
3: -59.1215591, 76.0324402, -59.3768349, 76.4206467, -135.5422058, 135.4092712
4: -54.1838837, 76.2210464, -54.4020615, 76.6039352, -130.7878113, 130.6230927

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_A2_B2_B2_A1

### Relational analysis result of NS_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5506757, upper bound: 96.5508472
time: 0.96 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2

### Relational analysis result of NS_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6091266, upper bound: 96.6107613
time: 0.91 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.76 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 4, lower bound: -96.5203875, upper bound: 96.5750033
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 4, lower bound: -96.5276922, upper bound: 96.5774402
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 4, lower bound: -96.5203875, upper bound: 96.5787537
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 4, lower bound: -96.5276922, upper bound: 96.5862474
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 4, lower bound: -96.5783359, upper bound: 96.5498290
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 4, lower bound: -96.5783359, upper bound: 96.5499522
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 4, lower bound: -96.5885935, upper bound: 96.6007448
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 4, lower bound: -96.5885935, upper bound: 96.6007448
NS_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 4, lower bound: -96.5165007, upper bound: 96.5810424
NS_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 4, lower bound: -96.5165007, upper bound: 96.5940222
NS_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 4, lower bound: -96.5809789, upper bound: 96.5278624
NS_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 4, lower bound: -96.5809789, upper bound: 96.5291289
NS_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 4, lower bound: -96.5809792, upper bound: 96.5923412
NS_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 4, lower bound: -96.5809792, upper bound: 96.6027441
NS_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 4, lower bound: -96.5209323, upper bound: 96.5694693
NS_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 4, lower bound: -96.5175975, upper bound: 96.5573705
NS_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 4, lower bound: -96.5273141, upper bound: 96.5736455
NS_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 4, lower bound: -96.5273141, upper bound: 96.5775801
NS_A2_A1_B2_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.76
Output dim: 4, lower bound: -96.5136112, upper bound: 96.5121750
NS_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 4, lower bound: -96.5097627, upper bound: 96.5934604
NS_A2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 4, lower bound: -96.5702302, upper bound: 96.5006154
NS_A2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 4, lower bound: -96.5862742, upper bound: 96.5301141
NS_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 4, lower bound: -96.5853497, upper bound: 96.5031326
NS_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 4, lower bound: -96.5940258, upper bound: 96.5308331
NS_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 4, lower bound: -96.5977133, upper bound: 96.5862629
NS_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 4, lower bound: -96.5976149, upper bound: 96.5869317
NS_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 4, lower bound: -96.5506757, upper bound: 96.5508472
NS_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 4, lower bound: -96.6091266, upper bound: 96.6107613

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -20.7903099, 40.7841949, -35.1516113, 59.2091713, -79.9994812, 75.9358063
1: -22.8470573, 33.4906540, -38.4155922, 51.0388718, -73.8859177, 71.9062500
2: -23.4349899, 33.1973572, -39.3333931, 50.8798904, -74.3148727, 72.5307465
3: -27.4004288, 38.6741219, -45.2213821, 59.2580223, -86.6584473, 83.8955078
4: -26.3936100, 37.8679123, -41.8584366, 58.9250221, -85.3186340, 79.7263336

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4929376, upper bound: 96.5328041
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B1_A1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5160750, upper bound: 96.5724740
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5204525, upper bound: 96.5750033
time: 1.81 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5153770, upper bound: 96.5529102
time: 1.00 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -22.8262177, 42.5750351, -34.8124886, 58.5398026, -81.3660126, 77.3875198
1: -24.9886570, 35.5293121, -38.0380783, 50.5008888, -75.4895325, 73.5673904
2: -25.6622066, 35.3047295, -38.9492950, 50.3483047, -76.0105133, 74.2540283
3: -29.8193817, 41.1082802, -44.7737503, 58.6362190, -88.4555740, 85.8820267
4: -28.4289017, 40.3929825, -41.4452286, 58.3029480, -86.7318497, 81.8382111

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5140391, upper bound: 96.5502163
time: 0.93 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5194623, upper bound: 96.5741375
time: 1.12 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -20.7903099, 40.7841949, -34.1786346, 57.5382462, -78.3285522, 74.9628296
1: -22.8470573, 33.4906540, -37.3464203, 49.3948364, -72.2418747, 70.8370667
2: -23.4349899, 33.1973572, -38.2511826, 49.2476044, -72.6825790, 71.4485397
3: -27.4004288, 38.6741219, -43.9584045, 57.4022827, -84.8027039, 82.6325226
4: -26.3936100, 37.8679123, -40.7358665, 57.0290222, -83.4226303, 78.6037674

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4896977, upper bound: 96.5259916
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B2_A1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5173734, upper bound: 96.5780247
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5163976, upper bound: 96.5705491
time: 1.04 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5163976, upper bound: 96.5787537
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -22.8262177, 42.5750351, -33.8836441, 56.9592171, -79.7854309, 76.4586716
1: -24.9886570, 35.5293121, -37.0225983, 48.9295616, -73.9182129, 72.5519104
2: -25.6622066, 35.3047295, -37.9190979, 48.7860641, -74.4482727, 73.2238235
3: -29.8193817, 41.1082802, -43.5768471, 56.8636208, -86.6829910, 84.6851273
4: -28.4289017, 40.3929825, -40.3797379, 56.4963570, -84.9252472, 80.7727203

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5109521, upper bound: 96.5371270
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5205675, upper bound: 96.5839517
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -46.2481194, 78.3501587, -34.3436432, 58.0345650, -104.2826843, 112.6937943
1: -50.5733986, 66.4516220, -37.5347443, 49.8279419, -100.4013367, 103.9863586
2: -51.7981987, 66.5858612, -38.4794235, 49.7207184, -101.5189209, 105.0652847
3: -59.6356201, 76.9967270, -44.2864571, 57.8407898, -117.4764099, 121.2831726
4: -54.5897675, 77.1993713, -40.9803696, 57.5617218, -112.1514893, 118.1797409

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5783359, upper bound: 96.5498290
time: 0.92 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5783359, upper bound: 96.5498290
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -46.7881012, 78.8563232, -34.3436432, 58.0345650, -104.8226624, 113.1999664
1: -51.1504440, 67.0032349, -37.5347443, 49.8279419, -100.9783707, 104.5379791
2: -52.3843880, 67.1529312, -38.4794235, 49.7207184, -102.1051025, 105.6323547
3: -60.2749329, 77.6449966, -44.2864571, 57.8407898, -118.1157227, 121.9314575
4: -55.1617622, 77.8798294, -40.9803696, 57.5617218, -112.7234802, 118.8601990

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5783359, upper bound: 96.5498290
time: 1.02 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5783359, upper bound: 96.5499522
time: 0.61 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -46.2481194, 78.3501587, -35.9303055, 60.5144310, -106.7625504, 114.2804489
1: -50.5733986, 66.4516220, -39.2484093, 51.9334297, -102.5068130, 105.7000275
2: -51.7981987, 66.5858612, -40.2091751, 51.8252335, -103.6234207, 106.7950363
3: -59.6356201, 76.9967270, -46.3102989, 60.2947693, -119.9303894, 123.3070145
4: -54.5897675, 77.1993713, -42.7250977, 60.0578003, -114.6475525, 119.9244690

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5868446, upper bound: 96.5868396
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5868446, upper bound: 96.6007448
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -46.7881012, 78.8563232, -35.9303055, 60.5144310, -107.3025360, 114.7866211
1: -51.1504440, 67.0032349, -39.2484093, 51.9334297, -103.0838547, 106.2516479
2: -52.3843880, 67.1529312, -40.2091751, 51.8252335, -104.2096252, 107.3621063
3: -60.2749329, 77.6449966, -46.3102989, 60.2947693, -120.5696945, 123.9552689
4: -55.1617622, 77.8798294, -42.7250977, 60.0578003, -115.2195587, 120.6049271

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5868446, upper bound: 96.5868396
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5868446, upper bound: 96.6008679
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -22.5888329, 43.2161484, -46.7894211, 78.8589554, -101.4477615, 90.0055542
1: -24.7766838, 35.7365608, -51.1519814, 67.0058212, -91.7825012, 86.8885422
2: -25.4296799, 35.4515381, -52.3858948, 67.1554642, -92.5851440, 87.8374252
3: -29.6232643, 41.3063087, -60.2768440, 77.6478806, -107.2711258, 101.5831528
4: -28.3481541, 40.5114021, -55.1636429, 77.8824539, -106.2306061, 95.6750488

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.4421439, upper bound: 96.5033516
time: 0.79 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5278396, upper bound: 96.5808597
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -22.5888329, 43.2161484, -46.0311928, 77.7005310, -100.2893524, 89.2473297
1: -24.7766838, 35.7365608, -50.3242874, 65.8815689, -90.6582489, 86.0608521
2: -25.4296799, 35.4515381, -51.5641251, 66.0073242, -91.4370041, 87.0156555
3: -29.6232643, 41.3063087, -59.3308029, 76.3739243, -105.9971695, 100.6371155
4: -28.3481541, 40.5114021, -54.3605728, 76.5541840, -104.9023361, 94.8719711

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.4421439, upper bound: 96.5033516
time: 1.11 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5278396, upper bound: 96.5808597
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -46.8327942, 78.9024353, -22.5569458, 43.1898117, -90.0226059, 101.4593658
1: -51.1982498, 67.0478745, -24.7445717, 35.7115326, -86.9097519, 91.7924194
2: -52.4324837, 67.1990433, -25.3966751, 35.4250870, -87.8575745, 92.5957184
3: -60.3272438, 77.6979752, -29.5906868, 41.2766342, -101.6038589, 107.2886505
4: -55.2086487, 77.9360428, -28.3221054, 40.4803848, -95.6890335, 106.2581482

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B1_B1_B1

### Relational analysis result of NS_A1_B2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5783055, upper bound: 96.5231909
time: 1.23 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B1_B1_B1

### Relational analysis result of NS_A1_B2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5793134, upper bound: 96.5211619
time: 0.83 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_B2

### Relational analysis result of NS_A1_B2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5809789, upper bound: 96.5278624
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -46.8327942, 78.9024353, -22.0632725, 42.4937210, -89.3265152, 100.9656982
1: -51.1982498, 67.0478745, -24.2183628, 35.0233841, -86.2216110, 91.2662277
2: -52.4324837, 67.1990433, -24.8598690, 34.6991234, -87.1316071, 92.0588913
3: -60.3272438, 77.6979752, -28.9810963, 40.5176315, -100.8448792, 106.6790695
4: -55.2086487, 77.9360428, -27.8329735, 39.6638184, -94.8724670, 105.7690125

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B1_B2_B1

### Relational analysis result of NS_A1_B2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5783055, upper bound: 96.5253936
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A2_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B1_B2_B1

### Relational analysis result of NS_A1_B2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5793134, upper bound: 96.5291289
time: 0.89 seconds

## Relational analysis of NS_A1_B2_A2_B1_B2_B2

### Relational analysis result of NS_A1_B2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5809789, upper bound: 96.5291101
time: 1.00 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -46.8327942, 78.9024353, -46.7894211, 78.8589554, -125.6917419, 125.6918335
1: -51.1982498, 67.0478745, -51.1519814, 67.0058212, -118.2040634, 118.1998520
2: -52.4324837, 67.1990433, -52.3858948, 67.1554642, -119.5879517, 119.5849380
3: -60.3272438, 77.6979752, -60.2768440, 77.6478806, -137.9751282, 137.9748230
4: -55.2086487, 77.9360428, -55.1636429, 77.8824539, -133.0910950, 133.0996857

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B2_B1_B1

### Relational analysis result of NS_A1_B2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5842411, upper bound: 96.5561741
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A2_B2_B1_B2

### Relational analysis result of NS_A1_B2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5899815, upper bound: 96.5899808
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -46.8327942, 78.9024353, -46.0311928, 77.7005310, -124.5333252, 124.9336243
1: -51.1982498, 67.0478745, -50.3242874, 65.8815689, -117.0798187, 117.3721619
2: -52.4324837, 67.1990433, -51.5641251, 66.0073242, -118.4398041, 118.7631607
3: -60.3272438, 77.6979752, -59.3308029, 76.3739243, -136.7011719, 137.0287781
4: -55.2086487, 77.9360428, -54.3605728, 76.5541840, -131.7628326, 132.2966156

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5549105, upper bound: 96.5680412
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A2_B2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5899814, upper bound: 96.6024385
time: 1.34 seconds

## BFS NS instance: NS_A2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -16.0013294, 33.7018280, -32.6261368, 55.1176834, -71.1190109, 66.3279648
1: -17.6864433, 27.2826900, -35.6369095, 47.4171524, -65.1035919, 62.9195938
2: -18.1214790, 26.9609604, -36.5287132, 47.2344055, -65.3558807, 63.4896698
3: -21.4284306, 31.4723530, -41.9983826, 55.0962029, -76.5246201, 73.4707336
4: -21.0283413, 30.6083851, -39.0019569, 54.6633034, -75.6916351, 69.6103363

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5196322, upper bound: 96.5691409
time: 0.80 seconds

## Relational analysis of NS_A2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5153845, upper bound: 96.5655608
time: 0.74 seconds

## BFS NS instance: NS_A2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -16.0013294, 33.7018280, -34.0349655, 57.4939651, -73.4952927, 67.7367783
1: -17.6864433, 27.2826900, -37.1853180, 49.3172760, -67.0037231, 64.4680023
2: -18.1214790, 26.9609604, -38.1041336, 49.1673508, -67.2888336, 65.0650940
3: -21.4284306, 31.4723530, -43.8030853, 57.3024864, -78.7308960, 75.2754364
4: -21.0283413, 30.6083851, -40.6121025, 56.9361763, -77.9645157, 71.2204819

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5162207, upper bound: 96.5570421
time: 0.73 seconds

## Relational analysis of NS_A2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5136274, upper bound: 96.5558823
time: 1.09 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -20.6750565, 40.6228981, -28.4493523, 48.9939651, -69.6690063, 69.0722504
1: -22.7359962, 33.3683281, -31.1174393, 41.8624573, -64.5984497, 64.4857635
2: -23.3511162, 33.0451317, -31.8602886, 41.6697350, -65.0208511, 64.9054184
3: -27.3136444, 38.5819588, -36.7477264, 48.5925064, -75.9061432, 75.3296814
4: -26.3192291, 37.7103271, -34.2325287, 48.1474953, -74.4667206, 71.9428558

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5273141, upper bound: 96.5712788
time: 0.80 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5272312, upper bound: 96.5696115
time: 1.11 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -20.6750565, 40.6228981, -34.3511200, 57.8638611, -78.5389175, 74.9740067
1: -22.7359962, 33.3683281, -37.5259399, 49.6403656, -72.3763504, 70.8942642
2: -23.3511162, 33.0451317, -38.4459763, 49.5317307, -72.8828430, 71.4911041
3: -27.3136444, 38.5819588, -44.1957207, 57.6732407, -84.9868774, 82.7776794
4: -26.3192291, 37.7103271, -40.8721161, 57.3466606, -83.6658859, 78.5824432

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5273141, upper bound: 96.5770405
time: 0.80 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5272312, upper bound: 96.5770241
time: 0.87 seconds

## BFS NS instance: NS_A2_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -21.9830303, 42.5499344, -47.7190552, 79.9503479, -101.9333725, 90.2689896
1: -24.1377525, 34.9816208, -52.1214371, 67.8727264, -92.0104446, 87.1030426
2: -24.7742939, 34.6529198, -53.4082642, 68.0354233, -92.8097153, 88.0611877
3: -28.8974457, 40.4633179, -61.3669090, 78.6607513, -107.5581970, 101.8302307
4: -27.7603989, 39.6223221, -56.2001648, 78.9533081, -106.7137070, 95.8224716

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_A1_B2_B2_A2_B1

### Relational analysis result of NS_A2_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5241733, upper bound: 96.5823636
time: 0.71 seconds

## Relational analysis of NS_A2_A1_B2_B2_A2_B2

### Relational analysis result of NS_A2_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5244653, upper bound: 96.5821969
time: 0.84 seconds

## BFS NS instance: NS_A2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -44.2931366, 75.5381470, -19.0168400, 37.8047943, -82.0979309, 94.5549850
1: -48.4466019, 63.8180428, -20.9385338, 31.1538448, -79.6004486, 84.7565765
2: -49.6492310, 63.9071274, -21.5286846, 30.8517284, -80.5009537, 85.4358139
3: -57.1849480, 73.9580307, -25.2066669, 36.0842285, -93.2691803, 99.1646957
4: -52.4466515, 74.0651245, -24.6514683, 35.1617165, -87.6083679, 98.7165756

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5537697, upper bound: 96.4946581
time: 0.88 seconds

## Relational analysis of NS_A2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5511382, upper bound: 96.4937422
time: 0.79 seconds

## BFS NS instance: NS_A2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -45.2401199, 76.6829147, -22.9655075, 43.7550316, -88.9951324, 99.6484222
1: -49.4706955, 64.9128571, -25.1754646, 36.1651459, -85.6358414, 90.0883026
2: -50.6889076, 65.0169983, -25.8641987, 35.8516693, -86.5405731, 90.8811798
3: -58.3554497, 75.2422791, -30.0730324, 41.8423233, -100.1977692, 105.3152924
4: -53.4867287, 75.3970261, -28.8824005, 41.0101280, -94.4968567, 104.2794113

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5726417, upper bound: 96.5270704
time: 0.98 seconds

## Relational analysis of NS_A2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5769755, upper bound: 96.5293950
time: 0.91 seconds

## BFS NS instance: NS_A2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -44.8781471, 76.0880127, -19.0168400, 37.8047943, -82.6829376, 95.1048508
1: -49.0697861, 64.4161682, -20.9385338, 31.1538448, -80.2236328, 85.3547058
2: -50.2835884, 64.5217667, -21.5286846, 30.8517284, -81.1353149, 86.0504532
3: -57.8710785, 74.6666031, -25.2066669, 36.0842285, -93.9553070, 99.8732681
4: -53.0723114, 74.8026199, -24.6514683, 35.1617165, -88.2340240, 99.4540710

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_A2_B1_A2_B1_B1

### Relational analysis result of NS_A2_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5831759, upper bound: 96.4958467
time: 0.76 seconds

## Relational analysis of NS_A2_A2_B1_A2_B1_B2

### Relational analysis result of NS_A2_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5837690, upper bound: 96.5008323
time: 0.75 seconds

## BFS NS instance: NS_A2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -45.8345718, 77.2436981, -22.9655075, 43.7550316, -89.5895844, 100.2092056
1: -50.1043701, 65.5200729, -25.1754646, 36.1651459, -86.2695084, 90.6955261
2: -51.3336945, 65.6416855, -25.8641987, 35.8516693, -87.1853638, 91.5058594
3: -59.0544815, 75.9624939, -30.0730324, 41.8423233, -100.8967972, 106.0355225
4: -54.1233749, 76.1475067, -28.8824005, 41.0101280, -95.1334991, 105.0298996

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_A2_B1_A2_B2_B1

### Relational analysis result of NS_A2_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5925077, upper bound: 96.5289563
time: 0.83 seconds

## Relational analysis of NS_A2_A2_B1_A2_B2_B2

### Relational analysis result of NS_A2_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5910074, upper bound: 96.5222856
time: 0.97 seconds

## BFS NS instance: NS_A2_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -40.1451073, 68.4505844, -45.6697693, 77.2373657, -117.3824692, 114.1203537
1: -43.8889236, 57.7715073, -49.9431610, 65.5871277, -109.4760437, 107.7146683
2: -44.9469833, 57.8029289, -51.1514740, 65.7121353, -110.6591187, 108.9543991
3: -51.7135468, 66.9439621, -58.8795891, 76.0140152, -127.7275620, 125.8235397
4: -47.5987549, 66.9218521, -53.9238510, 76.2002335, -123.7989731, 120.8457031

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_B2_B1_A1_B1

### Relational analysis result of NS_A2_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5965600, upper bound: 96.5860485
time: 0.76 seconds

## Relational analysis of NS_A2_A2_B2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_A2_B2_B1_A1_B1

### Relational analysis result of NS_A2_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5963961, upper bound: 96.5852329
time: 0.84 seconds

## Relational analysis of NS_A2_A2_B2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_A2_B2_B1_A1_A1

### Relational analysis result of NS_A2_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5921821, upper bound: 96.5636871
time: 0.83 seconds

## Relational analysis of NS_A2_A2_B2_B1_A1_A2

### Relational analysis result of NS_A2_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5795024, upper bound: 96.5591609
time: 0.84 seconds

## BFS NS instance: NS_A2_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -43.9369774, 74.7769775, -46.7962341, 78.8564758, -122.7934570, 121.5732117
1: -48.0688438, 63.2433891, -51.1590309, 67.0059967, -115.0748062, 114.4024200
2: -49.2526169, 63.3615570, -52.3925171, 67.1570892, -116.4097061, 115.7540741
3: -56.7578201, 73.2668686, -60.2832451, 77.6483307, -134.4061584, 133.5501099
4: -51.9256248, 73.4459763, -55.1672821, 77.8852844, -129.8109131, 128.6132355

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_A2_B2_B1_A2_A1

### Relational analysis result of NS_A2_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5865923, upper bound: 96.5369221
time: 1.52 seconds

## Relational analysis of NS_A2_A2_B2_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_A2_B2_B1_A2_A1

### Relational analysis result of NS_A2_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5498328, upper bound: 96.5500077
time: 0.94 seconds

## Relational analysis of NS_A2_A2_B2_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_A2_B2_B1_A2_B1

### Relational analysis result of NS_A2_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5924187, upper bound: 96.5662610
time: 0.83 seconds

## Relational analysis of NS_A2_A2_B2_B1_A2_B2

### Relational analysis result of NS_A2_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5967357, upper bound: 96.5858063
time: 0.95 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -44.1956596, 75.0196304, -46.0706635, 77.7413635, -121.9370270, 121.0902939
1: -48.3261871, 63.3727074, -50.3664093, 65.9208984, -114.2470856, 113.7390900
2: -49.4990959, 63.5028954, -51.6066208, 66.0480347, -115.5471344, 115.1095123
3: -56.9952431, 73.4271317, -59.3768349, 76.4206467, -133.4158783, 132.8039703
4: -52.2012939, 73.5888824, -54.4020615, 76.6039352, -128.8052216, 127.9909439

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_A2_B2_B2_A1_B1

### Relational analysis result of NS_A2_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5439481, upper bound: 96.5390784
time: 0.97 seconds

## Relational analysis of NS_A2_A2_B2_B2_A1_B2

### Relational analysis result of NS_A2_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5462027, upper bound: 96.5463635
time: 0.96 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -47.9622726, 80.4256058, -46.0480957, 77.7117386, -125.6740112, 126.4736786
1: -52.4078369, 68.5239410, -50.3421021, 65.8948975, -118.3027115, 118.8660431
2: -53.7301865, 68.6245117, -51.5818634, 66.0216446, -119.7518234, 120.2063751
3: -61.7738152, 79.3425293, -59.3494339, 76.3902817, -138.1640930, 138.6919556
4: -56.5845146, 79.5926056, -54.3775101, 76.5721130, -133.1566315, 133.9701080

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 14

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_A2_B2_B2_A2_B1

### Relational analysis result of NS_A2_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6033923, upper bound: 96.5584571
time: 1.18 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2_B2

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6033924, upper bound: 96.6107613
time: 0.97 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.10 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5204525, upper bound: 96.5750033
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5153770, upper bound: 96.5529102
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5140391, upper bound: 96.5502163
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5194623, upper bound: 96.5741375
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5163976, upper bound: 96.5705491
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5163976, upper bound: 96.5787537
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5109521, upper bound: 96.5371270
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5205675, upper bound: 96.5839517
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5783359, upper bound: 96.5498290
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5783359, upper bound: 96.5498290
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5783359, upper bound: 96.5498290
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5783359, upper bound: 96.5499522
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5868446, upper bound: 96.5868396
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5868446, upper bound: 96.6007448
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5868446, upper bound: 96.5868396
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5868446, upper bound: 96.6008679
NS_A1_B2_A1_B2_B1_A1, status: Status.VERIFIED, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.4421439, upper bound: 96.5033516
NS_A1_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5278396, upper bound: 96.5808597
NS_A1_B2_A1_B2_B2_A1, status: Status.VERIFIED, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.4421439, upper bound: 96.5033516
NS_A1_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5278396, upper bound: 96.5808597
NS_A1_B2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5793134, upper bound: 96.5211619
NS_A1_B2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5809789, upper bound: 96.5278624
NS_A1_B2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5793134, upper bound: 96.5291289
NS_A1_B2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5809789, upper bound: 96.5291101
NS_A1_B2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5842411, upper bound: 96.5561741
NS_A1_B2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5899815, upper bound: 96.5899808
NS_A1_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5549105, upper bound: 96.5680412
NS_A1_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5899814, upper bound: 96.6024385
NS_A2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5196322, upper bound: 96.5691409
NS_A2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5153845, upper bound: 96.5655608
NS_A2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5162207, upper bound: 96.5570421
NS_A2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5136274, upper bound: 96.5558823
NS_A2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5273141, upper bound: 96.5712788
NS_A2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5272312, upper bound: 96.5696115
NS_A2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5273141, upper bound: 96.5770405
NS_A2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5272312, upper bound: 96.5770241
NS_A2_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5241733, upper bound: 96.5823636
NS_A2_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5244653, upper bound: 96.5821969
NS_A2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5537697, upper bound: 96.4946581
NS_A2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5511382, upper bound: 96.4937422
NS_A2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5726417, upper bound: 96.5270704
NS_A2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5769755, upper bound: 96.5293950
NS_A2_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5831759, upper bound: 96.4958467
NS_A2_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5837690, upper bound: 96.5008323
NS_A2_A2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5925077, upper bound: 96.5289563
NS_A2_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5910074, upper bound: 96.5222856
NS_A2_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5921821, upper bound: 96.5636871
NS_A2_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5795024, upper bound: 96.5591609
NS_A2_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5924187, upper bound: 96.5662610
NS_A2_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5967357, upper bound: 96.5858063
NS_A2_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5439481, upper bound: 96.5390784
NS_A2_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.5462027, upper bound: 96.5463635
NS_A2_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.6033923, upper bound: 96.5584571
NS_A2_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 4, lower bound: -96.6033924, upper bound: 96.6107613

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -20.7903099, 40.7841949, -33.2582932, 56.3143501, -77.1046600, 74.0424881
1: -22.8470573, 33.4906540, -36.3492508, 48.6519814, -71.4990387, 69.8399048
2: -23.4349899, 33.1973572, -37.2379723, 48.4498024, -71.8847733, 70.4353180
3: -27.4004288, 38.6741219, -42.8507538, 56.4796562, -83.8800812, 81.5248718
4: -26.3936100, 37.8679123, -39.7421341, 56.0560112, -82.4496155, 77.6100311

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5193034, upper bound: 96.5745923
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5167110, upper bound: 96.5589712
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -20.7903099, 40.7841949, -34.7042656, 58.7233200, -79.5136261, 75.4884491
1: -22.8470573, 33.4906540, -37.9399986, 50.5811882, -73.4282455, 71.4306335
2: -23.4349899, 33.1973572, -38.8535500, 50.4153442, -73.8503265, 72.0509033
3: -27.4004288, 38.6741219, -44.7019234, 58.7213020, -86.1217117, 83.3760300
4: -26.3936100, 37.8679123, -41.3946724, 58.3706436, -84.7642517, 79.2625809

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5004294, upper bound: 96.4740544
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.4861320, upper bound: 96.4714099
time: 0.62 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -22.8262177, 42.5750351, -32.3719864, 55.4200859, -78.2463074, 74.9470139
1: -24.9886570, 35.5293121, -35.3950081, 47.5305672, -72.5192184, 70.9243164
2: -25.6622066, 35.3047295, -36.2755852, 47.3936996, -73.0559082, 71.5803070
3: -29.8193817, 41.1082802, -41.7917900, 55.1603966, -84.9797668, 82.9000702
4: -28.4289017, 40.3929825, -38.7316132, 54.7747498, -83.2036514, 79.1245956

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5139943, upper bound: 96.5502163
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5129427, upper bound: 96.5468446
time: 0.58 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -22.8262177, 42.5750351, -34.1141891, 58.2281303, -81.0543442, 76.6892242
1: -24.9886570, 35.5293121, -37.2998085, 49.9720840, -74.9607315, 72.8291168
2: -25.6622066, 35.3047295, -38.1899033, 49.8202629, -75.4824677, 73.4946289
3: -29.8193817, 41.1082802, -44.0650368, 57.9839745, -87.8033447, 85.1733170
4: -28.4289017, 40.3929825, -40.6931534, 57.6482658, -86.0771637, 81.0861206

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5146228, upper bound: 96.5716880
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5194381, upper bound: 96.5740403
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5132730, upper bound: 96.5501492
time: 0.87 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -20.7903099, 40.7841949, -32.0683517, 54.6495438, -75.4398422, 72.8525467
1: -22.8470573, 33.4906540, -35.0475082, 46.7532120, -69.6002426, 68.5381470
2: -23.4349899, 33.1973572, -35.9121208, 46.5810623, -70.0160446, 69.1094818
3: -27.4004288, 38.6741219, -41.3040161, 54.3262672, -81.7266846, 79.9781342
4: -26.3936100, 37.8679123, -38.3782883, 53.8706932, -80.2642975, 76.2461777

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5155974, upper bound: 96.5705491
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5163976, upper bound: 96.5661482
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -20.7903099, 40.7841949, -34.8494949, 57.5004158, -78.2907181, 75.6336899
1: -22.8470573, 33.4906540, -38.0462151, 49.7176170, -72.5646667, 71.5368652
2: -23.4349899, 33.1973572, -38.9640732, 49.6076355, -73.0426178, 72.1614304
3: -27.4004288, 38.6741219, -44.7448425, 57.8156242, -85.2160416, 83.4189606
4: -26.3936100, 37.8679123, -41.3713913, 57.5581169, -83.9517212, 79.2393036

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5155974, upper bound: 96.5787537
time: 0.91 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5163976, upper bound: 96.5661482
time: 0.86 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -22.8262177, 42.5750351, -31.5509682, 53.9670792, -76.7932968, 74.1260071
1: -24.9886570, 35.5293121, -34.4969254, 46.0989189, -71.0875626, 70.0262375
2: -25.6622066, 35.3047295, -35.3628044, 45.9668579, -71.6290665, 70.6675262
3: -29.8193817, 41.1082802, -40.7308121, 53.5500832, -83.3694458, 81.8390884
4: -28.4289017, 40.3929825, -37.7805634, 53.1334534, -81.5623550, 78.1735458

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5109521, upper bound: 96.5371270
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5109521, upper bound: 96.5371270
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -22.8262177, 42.5750351, -32.6441536, 55.7502861, -78.5764923, 75.2191925
1: -24.9886570, 35.5293121, -35.6867027, 47.6395988, -72.6282501, 71.2160187
2: -25.6622066, 35.3047295, -36.5559387, 47.4880981, -73.1502991, 71.8606567
3: -29.8193817, 41.1082802, -42.1690712, 55.3286667, -85.1480408, 83.2773438
4: -28.4289017, 40.3929825, -39.0143967, 54.9446411, -83.3735428, 79.4073792

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5157805, upper bound: 96.5826780
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5205675, upper bound: 96.5839517
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5205675, upper bound: 96.5839517
time: 0.88 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -46.2481194, 78.3501587, -33.7170753, 57.3991241, -103.6472397, 112.0672302
1: -50.5733986, 66.4516220, -36.8805351, 49.4016266, -99.9750214, 103.3321533
2: -51.7981987, 66.5858612, -37.7980652, 49.2694092, -101.0675964, 104.3839188
3: -59.6356201, 76.9967270, -43.5519142, 57.2936287, -116.9292297, 120.5486374
4: -54.5897675, 77.1993713, -40.3136482, 56.9988594, -111.5886230, 117.5130157

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5544146, upper bound: 96.5444110
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5544146, upper bound: 96.5498342
time: 0.84 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -46.2481194, 78.3501587, -32.9398384, 55.9766121, -102.2247314, 111.2899780
1: -50.5733986, 66.4516220, -36.0240326, 48.0106812, -98.5840607, 102.4756546
2: -51.7981987, 66.5858612, -36.9281311, 47.8837662, -99.6819611, 103.5139923
3: -59.6356201, 76.9967270, -42.5325165, 55.7216530, -115.3572693, 119.5292435
4: -54.5897675, 77.1993713, -39.3948784, 55.4065742, -109.9963379, 116.5942383

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5544146, upper bound: 96.5444110
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5544146, upper bound: 96.5498342
time: 0.88 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -46.7881012, 78.8563232, -33.7170753, 57.3991241, -104.1872253, 112.5733948
1: -51.1504440, 67.0032349, -36.8805351, 49.4016266, -100.5520706, 103.8837662
2: -52.3843880, 67.1529312, -37.7980652, 49.2694092, -101.6537933, 104.9509888
3: -60.2749329, 77.6449966, -43.5519142, 57.2936287, -117.5685501, 121.1969070
4: -55.1617622, 77.8798294, -40.3136482, 56.9988594, -112.1606216, 118.1934738

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5599584, upper bound: 96.5455223
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5599584, upper bound: 96.5499521
time: 3.46 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -46.7881012, 78.8563232, -32.9398384, 55.9766121, -102.7647095, 111.7961502
1: -51.1504440, 67.0032349, -36.0240326, 48.0106812, -99.1610947, 103.0272675
2: -52.3843880, 67.1529312, -36.9281311, 47.8837662, -100.2681580, 104.0810623
3: -60.2749329, 77.6449966, -42.5325165, 55.7216530, -115.9965820, 120.1775055
4: -55.1617622, 77.8798294, -39.3948784, 55.4065742, -110.5683365, 117.2746887

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5599584, upper bound: 96.5455223
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5599584, upper bound: 96.5499522
time: 0.87 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -46.2481194, 78.3501587, -35.5135574, 60.2043304, -106.4524460, 113.8637085
1: -50.5733986, 66.4516220, -38.8392754, 51.8292198, -102.4025955, 105.2908936
2: -51.7981987, 66.5858612, -39.7609940, 51.6851311, -103.4833145, 106.3468552
3: -59.6356201, 76.9967270, -45.8723030, 60.1126862, -119.7483063, 122.8690262
4: -54.5897675, 77.1993713, -42.3089447, 59.8713531, -114.4611053, 119.5083160

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5544146, upper bound: 96.5814216
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5544146, upper bound: 96.5868448
time: 1.14 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -46.2481194, 78.3501587, -34.2379265, 58.1512413, -104.3993607, 112.5880737
1: -50.5733986, 66.4516220, -37.4365273, 49.8464813, -100.4198608, 103.8881454
2: -51.7981987, 66.5858612, -38.3505554, 49.7083359, -101.5065308, 104.9364090
3: -59.6356201, 76.9967270, -44.2359123, 57.8597908, -117.4954071, 121.2326279
4: -54.5897675, 77.1993713, -40.8477364, 57.5693550, -112.1590958, 118.0471039

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5544146, upper bound: 96.5953267
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5544146, upper bound: 96.5892760
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -46.7881012, 78.8563232, -35.5135574, 60.2043304, -106.9924240, 114.3698807
1: -51.1504440, 67.0032349, -38.8392754, 51.8292198, -102.9796371, 105.8425140
2: -52.3843880, 67.1529312, -39.7609940, 51.6851311, -104.0695114, 106.9139252
3: -60.2749329, 77.6449966, -45.8723030, 60.1126862, -120.3876190, 123.5172958
4: -55.1617622, 77.8798294, -42.3089447, 59.8713531, -115.0331116, 120.1887741

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5599584, upper bound: 96.5825328
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5599584, upper bound: 96.5868396
time: 0.90 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -46.7881012, 78.8563232, -34.2379265, 58.1512413, -104.9393311, 113.0942230
1: -51.1504440, 67.0032349, -37.4365273, 49.8464813, -100.9968948, 104.4397583
2: -52.3843880, 67.1529312, -38.3505554, 49.7083359, -102.0927277, 105.5034790
3: -60.2749329, 77.6449966, -44.2359123, 57.8597908, -118.1347198, 121.8808899
4: -55.1617622, 77.8798294, -40.8477364, 57.5693550, -112.7311096, 118.7275696

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5599584, upper bound: 96.5964380
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5599584, upper bound: 96.5891890
time: 1.10 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -21.8014069, 42.2344170, -46.7641335, 78.8268280, -100.6282349, 88.9985428
1: -23.9376183, 34.8362427, -51.1248627, 66.9758835, -90.9134827, 85.9611053
2: -24.5745125, 34.5454941, -52.3583221, 67.1255417, -91.7000504, 86.9038162
3: -28.6815033, 40.2556305, -60.2462196, 77.6127548, -106.2942429, 100.5018463
4: -27.5317020, 39.4415665, -55.1355896, 77.8467484, -105.3784485, 94.5771484

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_B1_A2_A1

### Relational analysis result of NS_A1_B2_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4659417, upper bound: 96.5390787
time: 1.06 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_A2_A2

### Relational analysis result of NS_A1_B2_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5141669, upper bound: 96.5774349
time: 0.63 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -21.8014069, 42.2344170, -46.0131683, 77.6779251, -99.4793320, 88.2475891
1: -23.9376183, 34.8362427, -50.3049469, 65.8608398, -89.7984390, 85.1411896
2: -24.5745125, 34.5454941, -51.5445442, 65.9864197, -90.5609283, 86.0900269
3: -28.6815033, 40.2556305, -59.3089676, 76.3496323, -105.0311203, 99.5645981
4: -27.5317020, 39.4415665, -54.3409424, 76.5292664, -104.0609436, 93.7824936

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5012808, upper bound: 96.5402353
time: 0.89 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5012808, upper bound: 96.5936468
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -46.8327942, 78.9024353, -20.7579803, 40.7574387, -87.5902328, 99.6604004
1: -51.1982498, 67.0478745, -22.8144302, 33.4650574, -84.6632843, 89.8622971
2: -52.4324837, 67.1990433, -23.4014282, 33.1705093, -85.6029968, 90.6004715
3: -60.3272438, 77.6979752, -27.3671799, 38.6435280, -98.9707642, 105.0651474
4: -55.2086487, 77.9360428, -26.3670597, 37.8361359, -93.0447845, 104.3031006

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5445601, upper bound: 96.4948440
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_B1

### Relational analysis result of NS_A1_B2_A2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5725788, upper bound: 96.5169569
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_B1

### Relational analysis result of NS_A1_B2_A2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5228842, upper bound: 96.5095101
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_B1_B2

### Relational analysis result of NS_A1_B2_A2_B1_B1_B1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -96.5202396, upper bound: 96.4952123
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -46.4578934, 78.1730957, -22.7971821, 42.5501518, -89.0080414, 100.9702759
1: -50.7814522, 66.4642639, -24.9593277, 35.5056801, -86.2871246, 91.4235611
2: -52.0087204, 66.6214981, -25.6319809, 35.2799034, -87.2886200, 92.2534790
3: -59.8352814, 77.0206070, -29.7892952, 41.0800171, -100.9152985, 106.8099060
4: -54.7554207, 77.2595062, -28.4049320, 40.3637543, -95.1191711, 105.6644363

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_B1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5557597, upper bound: 96.5151490
time: 1.05 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5765439, upper bound: 96.5195789
time: 0.90 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -46.8327942, 78.9024353, -20.2902412, 40.0933647, -86.9261627, 99.1926575
1: -51.1982498, 67.0478745, -22.3122158, 32.8206406, -84.0188675, 89.3600845
2: -52.4324837, 67.1990433, -22.8950500, 32.4978294, -84.9303131, 90.0940857
3: -60.3272438, 77.6979752, -26.7845821, 37.9255905, -98.2528381, 104.4825592
4: -55.2086487, 77.9360428, -25.9135437, 37.0586090, -92.2672577, 103.8495865

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_B2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5559378, upper bound: 96.5189678
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A2_B1_B2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5767220, upper bound: 96.5234072
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -46.4578934, 78.1730957, -22.2411079, 41.7450333, -88.2029266, 100.4141998
1: -50.7814522, 66.4642639, -24.3782730, 34.7356186, -85.5170746, 90.8425293
2: -52.0087204, 66.6214981, -25.0326824, 34.4524002, -86.4611206, 91.6541824
3: -59.8352814, 77.0206070, -29.1233559, 40.2443314, -100.0796051, 106.1439667
4: -54.7554207, 77.2595062, -27.8707294, 39.4438171, -94.1992340, 105.1302261

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_B2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5559378, upper bound: 96.5180167
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A2_B1_B2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5767220, upper bound: 96.5224466
time: 0.56 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -46.8327942, 78.9024353, -45.0537415, 76.5162506, -123.3490448, 123.9561462
1: -51.1982498, 67.0478745, -49.2648544, 64.7434692, -115.9417114, 116.3127060
2: -52.4324837, 67.1990433, -50.4453087, 64.9047241, -117.3371964, 117.6443405
3: -60.3272438, 77.6979752, -58.0904617, 74.9719009, -135.2991486, 135.7884369
4: -55.2086487, 77.9360428, -53.1362000, 75.1825027, -130.3911285, 131.0722351

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B2_B1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5604885, upper bound: 96.5502011
time: 0.92 seconds

## Relational analysis of NS_A1_B2_A2_B2_B1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5819324, upper bound: 96.5548252
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -46.8139191, 78.8767700, -49.1458244, 82.4475708, -129.2614899, 128.0225983
1: -51.1779633, 67.0258713, -53.7041855, 70.3384628, -121.5164261, 120.7300415
2: -52.4116974, 67.1766968, -55.0324020, 70.4839630, -122.8956604, 122.2090988
3: -60.3043404, 77.6722336, -63.2940178, 81.3870010, -141.6913452, 140.9662476
4: -55.1880531, 77.9091339, -57.8911018, 81.7117691, -136.8998108, 135.8002014

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 14

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B2_B1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5549968, upper bound: 96.5605149
time: 0.87 seconds

## Relational analysis of NS_A1_B2_A2_B2_B1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5549968, upper bound: 96.5899808
time: 1.03 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -45.0967178, 76.5594177, -46.0311928, 77.7005310, -122.7972412, 122.5906067
1: -49.3106918, 64.7850266, -50.3242874, 65.8815689, -115.1922607, 115.1092987
2: -50.4913902, 64.9479218, -51.5641251, 66.0073242, -116.4986801, 116.5120468
3: -58.1404381, 75.0214539, -59.3308029, 76.3739243, -134.5143433, 134.3522644
4: -53.1804504, 75.2356949, -54.3605728, 76.5541840, -129.7346344, 129.5962524

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5240204, upper bound: 96.5400899
time: 0.96 seconds

## Relational analysis of NS_A1_B2_A2_B2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B2_B2_A1_A1

### Relational analysis result of NS_A1_B2_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5499519, upper bound: 96.5656939
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A2_B2_B2_A1_A2

### Relational analysis result of NS_A1_B2_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5502190, upper bound: 96.5658445
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -49.1832199, 82.4850388, -46.0085983, 77.6709061, -126.8541260, 128.4936218
1: -53.7440643, 70.3742752, -50.2999649, 65.8555527, -119.5996094, 120.6742325
2: -55.0724068, 70.5211716, -51.5393295, 65.9809189, -121.0533295, 122.0605011
3: -63.3375893, 81.4294510, -59.3033676, 76.3435059, -139.6810913, 140.7328186
4: -57.9294777, 81.7576675, -54.3359947, 76.5223389, -134.4517975, 136.0936432

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5842411, upper bound: 96.5561741
time: 0.79 seconds

## Relational analysis of NS_A1_B2_A2_B2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5842411, upper bound: 96.6024384
time: 0.94 seconds

## BFS NS instance: NS_A2_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -15.0919924, 32.5598259, -32.6261368, 55.1176834, -70.2096786, 65.1859589
1: -16.7231903, 26.2531586, -35.6369095, 47.4171524, -64.1403427, 61.8900681
2: -17.1289368, 25.9333820, -36.5287132, 47.2344055, -64.3633423, 62.4620972
3: -20.3348198, 30.2604980, -41.9983826, 55.0962029, -75.4310150, 72.2588806
4: -20.1022758, 29.3916588, -39.0019569, 54.6633034, -74.7655792, 68.3936157

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_A1_B1_A1_B1_A1_A1

### Relational analysis result of NS_A2_A1_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5170187, upper bound: 96.5683749
time: 0.87 seconds

## Relational analysis of NS_A2_A1_B1_A1_B1_A1_A2

### Relational analysis result of NS_A2_A1_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5148206, upper bound: 96.5663997
time: 0.88 seconds

## BFS NS instance: NS_A2_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -16.8622589, 35.2073593, -32.6261368, 55.1176834, -71.9799423, 67.8334961
1: -18.6416950, 28.5593109, -35.6369095, 47.4171524, -66.0588455, 64.1962128
2: -19.0992470, 28.2330036, -36.5287132, 47.2344055, -66.3336487, 64.7617111
3: -22.5876942, 32.9618950, -41.9983826, 55.0962029, -77.6838989, 74.9602737
4: -22.0284996, 32.1217384, -39.0019569, 54.6633034, -76.6918030, 71.1236954

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5153845, upper bound: 96.5655608
time: 0.80 seconds

## Relational analysis of NS_A2_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5153845, upper bound: 96.5655608
time: 0.77 seconds

## BFS NS instance: NS_A2_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -15.0919924, 32.5598259, -34.0349655, 57.4939651, -72.5859604, 66.5947876
1: -16.7231903, 26.2531586, -37.1853180, 49.3172760, -66.0404587, 63.4384766
2: -17.1289368, 25.9333820, -38.1041336, 49.1673508, -66.2962875, 64.0375137
3: -20.3348198, 30.2604980, -43.8030853, 57.3024864, -77.6372910, 74.0635834
4: -20.1022758, 29.3916588, -40.6121025, 56.9361763, -77.0384521, 70.0037537

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_A1_B1_A1_B2_A1_A1

### Relational analysis result of NS_A2_A1_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5133741, upper bound: 96.5562607
time: 0.70 seconds

## Relational analysis of NS_A2_A1_B1_A1_B2_A1_A2

### Relational analysis result of NS_A2_A1_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5121938, upper bound: 96.5558800
time: 0.94 seconds

## BFS NS instance: NS_A2_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -16.8622589, 35.2073593, -34.0349655, 57.4939651, -74.3562241, 69.2423172
1: -18.6416950, 28.5593109, -37.1853180, 49.3172760, -67.9589691, 65.7446289
2: -19.0992470, 28.2330036, -38.1041336, 49.1673508, -68.2666016, 66.3371277
3: -22.5876942, 32.9618950, -43.8030853, 57.3024864, -79.8901749, 76.7649841
4: -22.0284996, 32.1217384, -40.6121025, 56.9361763, -78.9646759, 72.7338409

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5069486, upper bound: 96.5368236
time: 0.69 seconds

## Relational analysis of NS_A2_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5069486, upper bound: 96.5558823
time: 0.98 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -18.8207550, 38.1322098, -28.4493523, 48.9939651, -67.8147202, 66.5815582
1: -20.7414513, 31.0542412, -31.1174393, 41.8624573, -62.6039085, 62.1716805
2: -21.2902985, 30.7325554, -31.8602886, 41.6697350, -62.9600334, 62.5928421
3: -25.0101185, 35.8563080, -36.7477264, 48.5925064, -73.6026230, 72.6040344
4: -24.2974548, 34.9831238, -34.2325287, 48.1474953, -72.4449463, 69.2156525

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5272690, upper bound: 96.5712788
time: 1.09 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_A1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5250041, upper bound: 96.5709650
time: 0.82 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_A2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5101766, upper bound: 96.5647377
time: 0.83 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -20.8937302, 39.8947296, -28.1516647, 48.4053879, -69.2991104, 68.0463867
1: -22.9341908, 33.0979843, -30.7905140, 41.3901596, -64.3243484, 63.8884964
2: -23.5599442, 32.8289032, -31.5252342, 41.2014923, -64.7614365, 64.3541412
3: -27.4981537, 38.3209953, -36.3626328, 48.0470314, -75.5451813, 74.6836166
4: -26.3660660, 37.5236015, -33.8753357, 47.6036339, -73.9696960, 71.3989410

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5272021, upper bound: 96.5696115
time: 0.92 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5266658, upper bound: 96.5677184
time: 0.81 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5266658, upper bound: 96.5696115
time: 0.79 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -18.8207550, 38.1322098, -34.3511200, 57.8638611, -76.6846161, 72.4833298
1: -20.7414513, 31.0542412, -37.5259399, 49.6403656, -70.3818130, 68.5801773
2: -21.2902985, 30.7325554, -38.4459763, 49.5317307, -70.8220291, 69.1785278
3: -25.0101185, 35.8563080, -44.1957207, 57.6732407, -82.6833572, 80.0520325
4: -24.2974548, 34.9831238, -40.8721161, 57.3466606, -81.6441193, 75.8552322

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_A1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5259054, upper bound: 96.5762438
time: 0.88 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_A2

### Relational analysis result of NS_A2_A1_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5095257, upper bound: 96.5647938
time: 0.77 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -20.8937302, 39.8947296, -34.0980072, 57.3332596, -78.2269745, 73.9927368
1: -22.9341908, 33.0979843, -37.2463913, 49.2248154, -72.1590042, 70.3443756
2: -23.5599442, 32.8289032, -38.1587219, 49.1212196, -72.6811676, 70.9876099
3: -27.4981537, 38.3209953, -43.8631401, 57.1927032, -84.6908569, 82.1841125
4: -26.3660660, 37.5236015, -40.5618210, 56.8697853, -83.2358551, 78.0854187

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5133608, upper bound: 96.5343073
time: 0.91 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5230297, upper bound: 96.5750077
time: 0.94 seconds

## BFS NS instance: NS_A2_A1_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -20.8526325, 40.9977493, -41.4901619, 70.2804794, -91.1330948, 82.4879150
1: -22.9228249, 33.5842476, -45.3206100, 59.3773613, -82.3001862, 78.9048462
2: -23.5280628, 33.2505226, -46.4202805, 59.4345970, -82.9626617, 79.6707916
3: -27.5088501, 38.8265190, -53.3425140, 68.7938156, -96.3026428, 92.1690292
4: -26.5066280, 37.9770699, -49.0683098, 68.8546600, -95.3612900, 87.0453796

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B2_B2_A2_B1_B1

### Relational analysis result of NS_A2_A1_B2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5086335, upper bound: 96.5471663
time: 0.59 seconds

## Relational analysis of NS_A2_A1_B2_B2_A2_B1_B2

### Relational analysis result of NS_A2_A1_B2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5210106, upper bound: 96.5800176
time: 0.76 seconds

## BFS NS instance: NS_A2_A1_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -21.9566402, 42.5154648, -45.7679825, 77.4062424, -99.3628845, 88.2834320
1: -24.1096935, 34.9504738, -50.0280113, 65.5433350, -89.6530151, 84.9784775
2: -24.7456055, 34.6216888, -51.2725220, 65.7026978, -90.4482956, 85.8942032
3: -28.8659058, 40.4267998, -59.0105782, 75.9060974, -104.7719955, 99.4373779
4: -27.7315331, 39.5857162, -53.9547539, 76.1816177, -103.9131470, 93.5404587

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B2_B2_A2_B2_A1

### Relational analysis result of NS_A2_A1_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4944759, upper bound: 96.5662474
time: 0.81 seconds

## Relational analysis of NS_A2_A1_B2_B2_A2_B2_A2

### Relational analysis result of NS_A2_A1_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5214206, upper bound: 96.5798510
time: 1.03 seconds

## BFS NS instance: NS_A2_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -38.5436935, 66.6829453, -17.8569355, 36.1792450, -74.7229309, 84.5398788
1: -42.1658592, 56.0084572, -19.6943836, 29.6945076, -71.8603668, 75.7028275
2: -43.1942215, 56.0091743, -20.2438278, 29.3863640, -72.5805817, 76.2529984
3: -49.7669983, 64.8621368, -23.7811069, 34.3780365, -84.1450195, 88.6432190
4: -45.8756294, 64.7601013, -23.3646202, 33.4533234, -79.3289490, 88.1247025

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A2_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_A2_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_A2_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5536942, upper bound: 96.4944436
time: 0.94 seconds

## Relational analysis of NS_A2_A2_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_A2_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5501840, upper bound: 96.4937422
time: 0.97 seconds

## Relational analysis of NS_A2_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5501840, upper bound: 96.4937422
time: 0.63 seconds

## BFS NS instance: NS_A2_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -42.3866158, 73.0618210, -18.9976883, 37.7805252, -80.1671448, 92.0594940
1: -46.4015274, 61.5346146, -20.9181061, 31.1320076, -77.5335388, 82.4527206
2: -47.5599518, 61.6195679, -21.5080643, 30.8300018, -78.3899460, 83.1276321
3: -54.8789978, 71.2595749, -25.1838341, 36.0585518, -90.9375381, 96.4434052
4: -50.2492332, 71.3527069, -24.6306038, 35.1360207, -85.3852539, 95.9833069

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5501840, upper bound: 96.4937422
time: 0.97 seconds

## Relational analysis of NS_A2_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5501840, upper bound: 96.4937422
time: 0.81 seconds

## BFS NS instance: NS_A2_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -39.4993935, 67.8216782, -21.6580811, 42.0505295, -81.5499268, 89.4797592
1: -43.1990509, 57.1039963, -23.7790604, 34.6091156, -77.8081665, 80.8830414
2: -44.2457466, 57.1161270, -24.4278526, 34.2856293, -78.5313568, 81.5439758
3: -50.9514923, 66.1553955, -28.4908524, 40.0159454, -90.9674377, 94.6462326
4: -46.9034576, 66.0976410, -27.4535103, 39.1768074, -86.0802612, 93.5511475

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5657372, upper bound: 96.5146708
time: 0.87 seconds

## Relational analysis of NS_A2_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5657369, upper bound: 96.5270704
time: 0.67 seconds

## BFS NS instance: NS_A2_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -43.2849579, 74.1491776, -22.9410629, 43.7239761, -87.0089340, 97.0902405
1: -47.3733749, 62.5743561, -25.1495533, 36.1376266, -83.5109863, 87.7238922
2: -48.5454712, 62.6737976, -25.8378754, 35.8240547, -84.3695221, 88.5116730
3: -55.9913025, 72.4741592, -30.0444126, 41.8100967, -97.8013840, 102.5185699
4: -51.2309074, 72.6183548, -28.8566513, 40.9775238, -92.2084351, 101.4750061

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5657369, upper bound: 96.5146708
time: 0.68 seconds

## Relational analysis of NS_A2_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5657369, upper bound: 96.5293950
time: 0.61 seconds

## BFS NS instance: NS_A2_A2_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -44.8781471, 76.0880127, -18.8045158, 37.6210060, -82.4991531, 94.8925323
1: -49.0697861, 64.4161682, -20.7213821, 30.9791679, -80.0489502, 85.1375504
2: -50.2835884, 64.5217667, -21.3083229, 30.6723366, -80.9559174, 85.8300934
3: -57.8710785, 74.6666031, -24.9816628, 35.8777466, -93.7488251, 99.6482697
4: -53.0723114, 74.8026199, -24.4570980, 34.9476204, -88.0199280, 99.2597198

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_A2_B1_A2_B1_B1_B1

### Relational analysis result of NS_A2_A2_B1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5580953, upper bound: 96.4467888
time: 0.60 seconds

## Relational analysis of NS_A2_A2_B1_A2_B1_B1_B2

### Relational analysis result of NS_A2_A2_B1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5819133, upper bound: 96.4911781
time: 0.96 seconds

## BFS NS instance: NS_A2_A2_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -44.8781471, 76.0880127, -18.8575993, 37.7701225, -82.6482620, 94.9456100
1: -49.0697861, 64.4161682, -20.7793446, 31.0269260, -80.0967102, 85.1955032
2: -50.2835884, 64.5217667, -21.3576927, 30.7190914, -81.0026779, 85.8794556
3: -57.8710785, 74.6666031, -25.0427723, 35.9291649, -93.8002472, 99.7093735
4: -53.0723114, 74.8026199, -24.5008812, 35.0223236, -88.0946274, 99.3034973

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_A2_B1_A2_B1_B2_B1

### Relational analysis result of NS_A2_A2_B1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5629564, upper bound: 96.4658202
time: 0.92 seconds

## Relational analysis of NS_A2_A2_B1_A2_B1_B2_B2

### Relational analysis result of NS_A2_A2_B1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5825024, upper bound: 96.4964677
time: 0.73 seconds

## BFS NS instance: NS_A2_A2_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -45.8345718, 77.2436981, -22.7174454, 43.5366402, -89.3712006, 99.9611435
1: -50.1043701, 65.5200729, -24.9226799, 35.9610977, -86.0654602, 90.4427414
2: -51.3336945, 65.6416855, -25.6056290, 35.6386604, -86.9723511, 91.2473145
3: -59.0544815, 75.9624939, -29.8132019, 41.6020737, -100.6565399, 105.7756958
4: -54.1233749, 76.1475067, -28.6601696, 40.7584915, -94.8818665, 104.8076782

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A2_B1_A2_B2_B1_A1

### Relational analysis result of NS_A2_A2_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5425758, upper bound: 96.5141980
time: 0.91 seconds

## Relational analysis of NS_A2_A2_B1_A2_B2_B1_A2

### Relational analysis result of NS_A2_A2_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5425758, upper bound: 96.5289563
time: 1.07 seconds

## BFS NS instance: NS_A2_A2_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -45.8345718, 77.2436981, -22.8106346, 43.7476273, -89.5821838, 100.0543289
1: -50.1043701, 65.5200729, -25.0190315, 36.0611420, -86.1654968, 90.5391083
2: -51.3336945, 65.6416855, -25.7005367, 35.7415085, -87.0752029, 91.3422165
3: -59.0544815, 75.9624939, -29.9134541, 41.7111130, -100.7655869, 105.8759460
4: -54.1233749, 76.1475067, -28.7407036, 40.8937531, -95.0171280, 104.8882141

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A2_B1_A2_B2_B2_A1

### Relational analysis result of NS_A2_A2_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5432174, upper bound: 96.5087256
time: 0.65 seconds

## Relational analysis of NS_A2_A2_B1_A2_B2_B2_A2

### Relational analysis result of NS_A2_A2_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5432174, upper bound: 96.4887641
time: 0.76 seconds

## BFS NS instance: NS_A2_A2_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -38.0880966, 65.4326782, -45.6697693, 77.2373657, -115.3254623, 111.1024475
1: -41.6657372, 55.2772675, -49.9431610, 65.5871277, -107.2528534, 105.2204285
2: -42.6922989, 55.2737579, -51.1514740, 65.7121353, -108.4044342, 106.4252167
3: -49.2029800, 64.0332718, -58.8795891, 76.0140152, -125.2169876, 122.9128342
4: -45.3766747, 63.9146881, -53.9238510, 76.2002335, -121.5769043, 117.8385391

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_A2_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A2_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B2_B1_A1_A1_A1

### Relational analysis result of NS_A2_A2_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5920040, upper bound: 96.5636871
time: 0.85 seconds

## Relational analysis of NS_A2_A2_B2_B1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_A2_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_A2_B2_B1_A1_A1_B1

### Relational analysis result of NS_A2_A2_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5795024, upper bound: 96.5591609
time: 0.85 seconds

## Relational analysis of NS_A2_A2_B2_B1_A1_A1_B2

### Relational analysis result of NS_A2_A2_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5795024, upper bound: 96.5591609
time: 0.69 seconds

## BFS NS instance: NS_A2_A2_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -39.8248901, 68.0631790, -45.6697693, 77.2373657, -117.0622559, 113.7329483
1: -43.5429611, 57.3989983, -49.9431610, 65.5871277, -109.1300888, 107.3421631
2: -44.5956726, 57.4292641, -51.1514740, 65.7121353, -110.3078079, 108.5807266
3: -51.3214912, 66.5034485, -58.8795891, 76.0140152, -127.3355026, 125.3830414
4: -47.2474213, 66.4677582, -53.9238510, 76.2002335, -123.4476547, 120.3916092

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B2_B1_A1_A2_A1

### Relational analysis result of NS_A2_A2_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5793082, upper bound: 96.5591585
time: 1.00 seconds

## Relational analysis of NS_A2_A2_B2_B1_A1_A2_A2

### Relational analysis result of NS_A2_A2_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5587983, upper bound: 96.5541881
time: 1.01 seconds

## BFS NS instance: NS_A2_A2_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -43.9369774, 74.7769775, -44.4481468, 75.7729568, -119.7099228, 119.2251282
1: -48.0688438, 63.2433891, -48.6019974, 64.0744171, -112.1432343, 111.8453827
2: -49.2526169, 63.3615570, -49.8004074, 64.2402802, -113.4928970, 113.1619644
3: -56.7578201, 73.2668686, -57.3635559, 74.1907883, -130.9485931, 130.6304321
4: -51.9256248, 73.4459763, -52.4957657, 74.3748245, -126.3004379, 125.9417343

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 42

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_A2_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_A2_B2_B1_A2_B1_A1

### Relational analysis result of NS_A2_A2_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5534246, upper bound: 96.5570137
time: 1.52 seconds

## Relational analysis of NS_A2_A2_B2_B1_A2_B1_A2

### Relational analysis result of NS_A2_A2_B2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5534246, upper bound: 96.5570137
time: 0.86 seconds

## BFS NS instance: NS_A2_A2_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -43.9369774, 74.7769775, -45.9766541, 78.6902390, -122.6272125, 120.7536316
1: -48.0688438, 63.2433891, -50.3225784, 66.5914764, -114.6602936, 113.5659637
2: -49.2526169, 63.3615570, -51.5299606, 66.7141953, -115.9668121, 114.8915100
3: -56.7578201, 73.2668686, -59.5152435, 77.0927429, -133.8505554, 132.7821045
4: -51.9256248, 73.4459763, -54.3953094, 77.2847290, -129.2103577, 127.8412704

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_A2_B2_B1_A2_B2_A1

### Relational analysis result of NS_A2_A2_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5552725, upper bound: 96.5656778
time: 0.72 seconds

## Relational analysis of NS_A2_A2_B2_B1_A2_B2_A2

### Relational analysis result of NS_A2_A2_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5552725, upper bound: 96.5858063
time: 1.36 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -44.1956596, 75.0196304, -43.7911530, 74.7614212, -118.9570770, 118.8107834
1: -48.3261871, 63.3727074, -47.8972168, 63.1007462, -111.4269333, 111.2699203
2: -49.4990959, 63.5028954, -49.0905914, 63.2299843, -112.7290802, 112.5934906
3: -56.9952431, 73.4271317, -56.5668640, 73.0977478, -130.0929718, 129.9939880
4: -52.2012939, 73.5888824, -51.8095932, 73.2200623, -125.4213562, 125.3984756

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_A1

### Relational analysis result of NS_A2_A2_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5358677, upper bound: 96.5359805
time: 0.77 seconds

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_A2

### Relational analysis result of NS_A2_A2_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5358677, upper bound: 96.5390784
time: 0.82 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -44.1956596, 75.0196304, -44.8673706, 77.1363297, -121.3319855, 119.8869934
1: -48.3261871, 63.3727074, -49.1090584, 65.0707169, -113.3969040, 112.4817505
2: -49.4990959, 63.5028954, -50.3306122, 65.1608963, -114.6599884, 113.8335114
3: -56.9952431, 73.4271317, -58.1334267, 75.3590240, -132.3542175, 131.5605621
4: -52.2012939, 73.5888824, -53.2058487, 75.4903183, -127.6916122, 126.7947311

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_A2_B2_B2_A1_B2_A1

### Relational analysis result of NS_A2_A2_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5377283, upper bound: 96.5418835
time: 0.76 seconds

## Relational analysis of NS_A2_A2_B2_B2_A1_B2_A2

### Relational analysis result of NS_A2_A2_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5377282, upper bound: 96.5463635
time: 1.11 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -47.9622726, 80.4256058, -44.3655128, 75.4342270, -123.3964996, 124.7911224
1: -52.4078369, 68.5239410, -48.5170097, 63.7010727, -116.1089020, 117.0409546
2: -53.7301865, 68.6245117, -49.6997604, 63.8332062, -117.5633850, 118.3242722
3: -61.7738152, 79.3425293, -57.2372551, 73.7988205, -135.5726166, 136.5797882
4: -56.5845146, 79.5926056, -52.4086609, 73.9540558, -130.5385742, 132.0012665

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_A2_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_A2_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_A2_B2_B2_A2_B1_A1

### Relational analysis result of NS_A2_A2_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5525439, upper bound: 96.5457380
time: 1.12 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2_B1_A2

### Relational analysis result of NS_A2_A2_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6019733, upper bound: 96.5549692
time: 0.81 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -47.9622726, 80.4256058, -48.3011398, 81.2727432, -129.2350159, 128.7267456
1: -52.4078369, 68.5239410, -52.7873878, 69.1760712, -121.5838928, 121.3113174
2: -53.7301865, 68.6245117, -54.1293259, 69.2777786, -123.0079498, 122.7538376
3: -61.7738152, 79.3425293, -62.2547340, 80.0733795, -141.8471832, 141.5972595
4: -56.5845146, 79.5926056, -56.9884682, 80.3158188, -136.9003296, 136.5810699

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B1

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6008392, upper bound: 96.5551241
time: 0.81 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B2

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.6024480, upper bound: 96.6099114
time: 0.72 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 3.30 seconds
NS_A1_B1_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5193034, upper bound: 96.5745923
NS_A1_B1_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5167110, upper bound: 96.5589712
NS_A1_B1_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5004294, upper bound: 96.4740544
NS_A1_B1_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.4861320, upper bound: 96.4714099
NS_A1_B1_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5139943, upper bound: 96.5502163
NS_A1_B1_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5129427, upper bound: 96.5468446
NS_A1_B1_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5194381, upper bound: 96.5740403
NS_A1_B1_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5132730, upper bound: 96.5501492
NS_A1_B1_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5155974, upper bound: 96.5705491
NS_A1_B1_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5163976, upper bound: 96.5661482
NS_A1_B1_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5155974, upper bound: 96.5787537
NS_A1_B1_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5163976, upper bound: 96.5661482
NS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5109521, upper bound: 96.5371270
NS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5109521, upper bound: 96.5371270
NS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5205675, upper bound: 96.5839517
NS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5205675, upper bound: 96.5839517
NS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5544146, upper bound: 96.5444110
NS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5544146, upper bound: 96.5498342
NS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5544146, upper bound: 96.5444110
NS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5544146, upper bound: 96.5498342
NS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5599584, upper bound: 96.5455223
NS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5599584, upper bound: 96.5499521
NS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5599584, upper bound: 96.5455223
NS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5599584, upper bound: 96.5499522
NS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5544146, upper bound: 96.5814216
NS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5544146, upper bound: 96.5868448
NS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5544146, upper bound: 96.5953267
NS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5544146, upper bound: 96.5892760
NS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5599584, upper bound: 96.5825328
NS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5599584, upper bound: 96.5868396
NS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5599584, upper bound: 96.5964380
NS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5599584, upper bound: 96.5891890
NS_A1_B2_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.4659417, upper bound: 96.5390787
NS_A1_B2_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5141669, upper bound: 96.5774349
NS_A1_B2_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5012808, upper bound: 96.5402353
NS_A1_B2_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5012808, upper bound: 96.5936468
NS_A1_B2_A2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5228842, upper bound: 96.5095101
NS_A1_B2_A2_B1_B1_B1_B2, status: Status.VERIFIED, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5202396, upper bound: 96.4952123
NS_A1_B2_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5557597, upper bound: 96.5151490
NS_A1_B2_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5765439, upper bound: 96.5195789
NS_A1_B2_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5559378, upper bound: 96.5189678
NS_A1_B2_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5767220, upper bound: 96.5234072
NS_A1_B2_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5559378, upper bound: 96.5180167
NS_A1_B2_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5767220, upper bound: 96.5224466
NS_A1_B2_A2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5604885, upper bound: 96.5502011
NS_A1_B2_A2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5819324, upper bound: 96.5548252
NS_A1_B2_A2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5549968, upper bound: 96.5605149
NS_A1_B2_A2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5549968, upper bound: 96.5899808
NS_A1_B2_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5499519, upper bound: 96.5656939
NS_A1_B2_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5502190, upper bound: 96.5658445
NS_A1_B2_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5842411, upper bound: 96.5561741
NS_A1_B2_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5842411, upper bound: 96.6024384
NS_A2_A1_B1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5170187, upper bound: 96.5683749
NS_A2_A1_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5148206, upper bound: 96.5663997
NS_A2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5153845, upper bound: 96.5655608
NS_A2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5153845, upper bound: 96.5655608
NS_A2_A1_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5133741, upper bound: 96.5562607
NS_A2_A1_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5121938, upper bound: 96.5558800
NS_A2_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5069486, upper bound: 96.5368236
NS_A2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5069486, upper bound: 96.5558823
NS_A2_A1_B1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5250041, upper bound: 96.5709650
NS_A2_A1_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5101766, upper bound: 96.5647377
NS_A2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5266658, upper bound: 96.5677184
NS_A2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5266658, upper bound: 96.5696115
NS_A2_A1_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5259054, upper bound: 96.5762438
NS_A2_A1_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5095257, upper bound: 96.5647938
NS_A2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5133608, upper bound: 96.5343073
NS_A2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5230297, upper bound: 96.5750077
NS_A2_A1_B2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5086335, upper bound: 96.5471663
NS_A2_A1_B2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5210106, upper bound: 96.5800176
NS_A2_A1_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.4944759, upper bound: 96.5662474
NS_A2_A1_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5214206, upper bound: 96.5798510
NS_A2_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5501840, upper bound: 96.4937422
NS_A2_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5501840, upper bound: 96.4937422
NS_A2_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5501840, upper bound: 96.4937422
NS_A2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5501840, upper bound: 96.4937422
NS_A2_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5657372, upper bound: 96.5146708
NS_A2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5657369, upper bound: 96.5270704
NS_A2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5657369, upper bound: 96.5146708
NS_A2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5657369, upper bound: 96.5293950
NS_A2_A2_B1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5580953, upper bound: 96.4467888
NS_A2_A2_B1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5819133, upper bound: 96.4911781
NS_A2_A2_B1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5629564, upper bound: 96.4658202
NS_A2_A2_B1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5825024, upper bound: 96.4964677
NS_A2_A2_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5425758, upper bound: 96.5141980
NS_A2_A2_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5425758, upper bound: 96.5289563
NS_A2_A2_B1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5432174, upper bound: 96.5087256
NS_A2_A2_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5432174, upper bound: 96.4887641
NS_A2_A2_B2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5795024, upper bound: 96.5591609
NS_A2_A2_B2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5795024, upper bound: 96.5591609
NS_A2_A2_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5793082, upper bound: 96.5591585
NS_A2_A2_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5587983, upper bound: 96.5541881
NS_A2_A2_B2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5534246, upper bound: 96.5570137
NS_A2_A2_B2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5534246, upper bound: 96.5570137
NS_A2_A2_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5552725, upper bound: 96.5656778
NS_A2_A2_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5552725, upper bound: 96.5858063
NS_A2_A2_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5358677, upper bound: 96.5359805
NS_A2_A2_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5358677, upper bound: 96.5390784
NS_A2_A2_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5377283, upper bound: 96.5418835
NS_A2_A2_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5377282, upper bound: 96.5463635
NS_A2_A2_B2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.5525439, upper bound: 96.5457380
NS_A2_A2_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.6019733, upper bound: 96.5549692
NS_A2_A2_B2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.6008392, upper bound: 96.5551241
NS_A2_A2_B2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.30
Output dim: 4, lower bound: -96.6024480, upper bound: 96.6099114

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -20.7535973, 40.7284164, -33.1300507, 56.3201675, -77.0737610, 73.8584518
1: -22.8071156, 33.4395065, -36.2138786, 48.5294876, -71.3366013, 69.6533813
2: -23.3940086, 33.1464958, -37.1039085, 48.3365097, -71.7304993, 70.2504044
3: -27.3544807, 38.6141663, -42.7708054, 56.3187866, -83.6732635, 81.3849716
4: -26.3495903, 37.8094215, -39.5737419, 55.9589500, -82.3085251, 77.3831482

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -20.7903099, 40.7841949, -32.5881844, 55.3879547, -76.1782684, 73.3723755
1: -22.8470573, 33.4906540, -35.6184235, 47.7642097, -70.6112671, 69.1090775
2: -23.4349899, 33.1973572, -36.4852257, 47.5666008, -71.0015869, 69.6825867
3: -27.4004288, 38.6741219, -42.0053711, 55.4410324, -82.8414612, 80.6794891
4: -26.3936100, 37.8679123, -38.9568787, 55.0068665, -81.4004745, 76.8247910

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4973782, upper bound: 96.5454191
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -22.8262177, 42.5750351, -30.6950436, 52.7123375, -75.5385590, 73.2700806
1: -24.9886570, 35.5293121, -33.5560646, 45.3146324, -70.3032837, 69.0853729
2: -25.6622066, 35.3047295, -34.4062042, 45.1426697, -70.8048782, 69.7109375
3: -29.8193817, 41.1082802, -39.6614571, 52.5817108, -82.4010925, 80.7697296
4: -28.4289017, 40.3929825, -36.8219337, 52.1236343, -80.5525208, 77.2149124

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5139943, upper bound: 96.5502163
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5139943, upper bound: 96.5502163
time: 0.61 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -22.8262177, 42.5750351, -32.1176186, 55.0607529, -77.8869705, 74.6926575
1: -24.9886570, 35.5293121, -35.1159821, 47.1946297, -72.1832733, 70.6452942
2: -25.6622066, 35.3047295, -35.9922295, 47.0579491, -72.7201538, 71.2969589
3: -29.8193817, 41.1082802, -41.4679108, 54.7679062, -84.5872574, 82.5761871
4: -28.4289017, 40.3929825, -38.4392509, 54.3785896, -82.8074875, 78.8322296

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5129427, upper bound: 96.5468446
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5129427, upper bound: 96.5468446
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -22.8262177, 42.5750351, -32.4017181, 55.5839844, -78.4102020, 74.9767456
1: -24.9886570, 35.5293121, -35.4345779, 47.8044510, -72.7930832, 70.9638901
2: -25.6622066, 35.3047295, -36.2943306, 47.6177254, -73.2799225, 71.5990601
3: -29.8193817, 41.1082802, -41.9232445, 55.4554787, -85.2748337, 83.0315247
4: -28.4289017, 40.3929825, -38.7782478, 55.0447044, -83.4736023, 79.1712036

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5139943, upper bound: 96.5740403
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5139943, upper bound: 96.5506250
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -22.8262177, 42.5750351, -33.6324539, 57.6900902, -80.5163116, 76.2074890
1: -24.9886570, 35.5293121, -36.7863121, 49.4677963, -74.4564362, 72.3156281
2: -25.6622066, 35.3047295, -37.6703644, 49.3097038, -74.9719086, 72.9750824
3: -29.8193817, 41.1082802, -43.5020065, 57.3914833, -87.2108459, 84.6102753
4: -28.4289017, 40.3929825, -40.1848488, 57.0383492, -85.4672546, 80.5778046

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5129427, upper bound: 96.5501492
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5129427, upper bound: 96.5468446
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -20.7903099, 40.7841949, -30.1518478, 51.7098923, -72.5001984, 70.9360428
1: -22.8470573, 33.4906540, -32.9548759, 44.3283539, -67.1754074, 66.4455185
2: -23.4349899, 33.1973572, -33.7909927, 44.1121864, -67.5471725, 66.9883499
3: -27.4004288, 38.6741219, -38.9071693, 51.5047340, -78.9051590, 77.5812912
4: -26.3936100, 37.8679123, -36.2366600, 50.9599190, -77.3535156, 74.1045532

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_B1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5141999, upper bound: 96.5701333
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -20.7903099, 40.7841949, -31.7206841, 54.2786293, -75.0689316, 72.5048752
1: -22.8470573, 33.4906540, -34.6781235, 46.4052811, -69.2523193, 68.1687622
2: -23.4349899, 33.1973572, -35.5395737, 46.2286835, -69.6636658, 68.7369156
3: -27.4004288, 38.6741219, -40.9067764, 53.9174042, -81.3178253, 79.5809021
4: -26.3936100, 37.8679123, -38.0182076, 53.4509468, -79.8445587, 75.8861084

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5152152, upper bound: 96.5668289
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_B2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5125550, upper bound: 96.5524946
time: 0.61 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -20.7903099, 40.7841949, -32.8439713, 54.5251427, -75.3154449, 73.6281662
1: -22.8470573, 33.4906540, -35.8634453, 47.2311859, -70.0782471, 69.3540955
2: -23.4349899, 33.1973572, -36.7520828, 47.0778122, -70.5127869, 69.9494400
3: -27.4004288, 38.6741219, -42.2506905, 54.9172630, -82.3176804, 80.9248123
4: -26.3936100, 37.8679123, -39.1421432, 54.5709763, -80.9645767, 77.0100555

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_B1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4973589, upper bound: 96.5434378
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_B1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5192332, upper bound: 96.5783379
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_B1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5155778, upper bound: 96.5581840
time: 0.86 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -20.7903099, 40.7841949, -34.4210854, 57.0299187, -77.8202209, 75.2052765
1: -22.8470573, 33.4906540, -37.5894165, 49.2740631, -72.1211090, 71.0800476
2: -23.4349899, 33.1973572, -38.5030594, 49.1571922, -72.5921555, 71.7004166
3: -27.4004288, 38.6741219, -44.2414513, 57.2953224, -84.6957321, 82.9155731
4: -26.3936100, 37.8679123, -40.9225273, 57.0183830, -83.4119873, 78.7904358

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.4973219, upper bound: 96.5433774
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5154650, upper bound: 96.5657253
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_B2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5145461, upper bound: 96.5568935
time: 0.88 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -21.5127430, 40.7873344, -31.5509682, 53.9670792, -75.4798203, 72.3383026
1: -23.5656700, 33.8974648, -34.4969254, 46.0989189, -69.6645813, 68.3943863
2: -24.2192669, 33.6783485, -35.3628044, 45.9668579, -70.1861191, 69.0411377
3: -28.2059822, 39.1932602, -40.7308121, 53.5500832, -81.7560654, 79.9240723
4: -26.9879379, 38.4628525, -37.7805634, 53.1334534, -80.1213913, 76.2434158

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -22.8881760, 42.8530312, -31.5509682, 53.9670792, -76.8552551, 74.4039993
1: -25.0880909, 35.7457237, -34.4969254, 46.0989189, -71.1870041, 70.2426453
2: -25.7523994, 35.5105972, -35.3628044, 45.9668579, -71.7192535, 70.8733673
3: -30.0326767, 41.3473396, -40.7308121, 53.5500832, -83.5827484, 82.0781326
4: -28.6125183, 40.6500511, -37.7805634, 53.1334534, -81.7459717, 78.4306183

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -21.5127430, 40.7873344, -32.6441536, 55.7502861, -77.2630234, 73.4314880
1: -23.5656700, 33.8974648, -35.6867027, 47.6395988, -71.2052689, 69.5841522
2: -24.2192669, 33.6783485, -36.5559387, 47.4880981, -71.7073441, 70.2342758
3: -28.2059822, 39.1932602, -42.1690712, 55.3286667, -83.5346527, 81.3623276
4: -26.9879379, 38.4628525, -39.0143967, 54.9446411, -81.9325790, 77.4772491

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -22.8881760, 42.8530312, -32.6441536, 55.7502861, -78.6384430, 75.4971848
1: -25.0880909, 35.7457237, -35.6867027, 47.6395988, -72.7276917, 71.4324265
2: -25.7523994, 35.5105972, -36.5559387, 47.4880981, -73.2404938, 72.0665283
3: -30.0326767, 41.3473396, -42.1690712, 55.3286667, -85.3613434, 83.5163956
4: -28.6125183, 40.6500511, -39.0143967, 54.9446411, -83.5571594, 79.6644440

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -43.8946152, 75.2652435, -33.7170753, 57.3991241, -101.2937241, 108.9823151
1: -48.0108223, 63.5181313, -36.8805351, 49.4016266, -97.4124374, 100.3986588
2: -49.2012405, 63.6662178, -37.7980652, 49.2694092, -98.4706421, 101.4642639
3: -56.7119904, 73.5356216, -43.5519142, 57.2936287, -114.0056152, 117.0875320
4: -51.9156113, 73.6841812, -40.3136482, 56.9988594, -108.9144745, 113.9978333

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 12

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -45.3713455, 78.1234589, -33.7170753, 57.3991241, -102.7704697, 111.8405304
1: -49.6764984, 65.9766541, -36.8805351, 49.4016266, -99.0781174, 102.8571701
2: -50.8740158, 66.0815887, -37.7980652, 49.2694092, -100.1434174, 103.8796463
3: -58.8034401, 76.3701706, -43.5519142, 57.2936287, -116.0970535, 119.9220886
4: -53.7591057, 76.5252075, -40.3136482, 56.9988594, -110.7579498, 116.8388519

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 12

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -43.8946152, 75.2652435, -32.9398384, 55.9766121, -99.8712311, 108.2050781
1: -48.0108223, 63.5181313, -36.0240326, 48.0106812, -96.0214767, 99.5421600
2: -49.2012405, 63.6662178, -36.9281311, 47.8837662, -97.0850067, 100.5943375
3: -56.7119904, 73.5356216, -42.5325165, 55.7216530, -112.4336395, 116.0681381
4: -51.9156113, 73.6841812, -39.3948784, 55.4065742, -107.3221893, 113.0790558

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 12

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -45.3713455, 78.1234589, -32.9398384, 55.9766121, -101.3479614, 111.0632858
1: -49.6764984, 65.9766541, -36.0240326, 48.0106812, -97.6871490, 102.0006714
2: -50.8740158, 66.0815887, -36.9281311, 47.8837662, -98.7577820, 103.0097198
3: -58.8034401, 76.3701706, -42.5325165, 55.7216530, -114.5250931, 118.9026871
4: -53.7591057, 76.5252075, -39.3948784, 55.4065742, -109.1656723, 115.9200745

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 12

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -44.4287033, 75.7598953, -33.7170753, 57.3991241, -101.8278275, 109.4769745
1: -48.5812683, 64.0594254, -36.8805351, 49.4016266, -97.9828949, 100.9399490
2: -49.7800446, 64.2237701, -37.7980652, 49.2694092, -99.0494537, 102.0218353
3: -57.3418617, 74.1730194, -43.5519142, 57.2936287, -114.6354828, 117.7249298
4: -52.4777603, 74.3543930, -40.3136482, 56.9988594, -109.4766083, 114.6680374

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5579423, upper bound: 96.5568828
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5568905, upper bound: 96.5535111
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -45.9702988, 78.6909714, -33.7170753, 57.3991241, -103.3694229, 112.4080505
1: -50.3157768, 66.5897369, -36.8805351, 49.4016266, -99.7173920, 103.4702759
2: -51.5235291, 66.7111588, -37.7980652, 49.2694092, -100.7929230, 104.5092087
3: -59.5086784, 77.0904236, -43.5519142, 57.2936287, -116.8022766, 120.6423340
4: -54.3909187, 77.2805634, -40.3136482, 56.9988594, -111.3897781, 117.5942078

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5579423, upper bound: 96.5569713
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5568905, upper bound: 96.5535997
time: 0.75 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -44.4287033, 75.7598953, -32.9398384, 55.9766121, -100.4053192, 108.6997375
1: -48.5812683, 64.0594254, -36.0240326, 48.0106812, -96.5919418, 100.0834503
2: -49.7800446, 64.2237701, -36.9281311, 47.8837662, -97.6638107, 101.1519012
3: -57.3418617, 74.1730194, -42.5325165, 55.7216530, -113.0635147, 116.7055283
4: -52.4777603, 74.3543930, -39.3948784, 55.4065742, -107.8843307, 113.7492447

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 12

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -45.9702988, 78.6909714, -32.9398384, 55.9766121, -101.9469147, 111.6308060
1: -50.3157768, 66.5897369, -36.0240326, 48.0106812, -98.3264313, 102.6137695
2: -51.5235291, 66.7111588, -36.9281311, 47.8837662, -99.4072952, 103.6392746
3: -59.5086784, 77.0904236, -42.5325165, 55.7216530, -115.2303162, 119.6229401
4: -54.3909187, 77.2805634, -39.3948784, 55.4065742, -109.7974930, 116.6754303

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 12

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -43.8946152, 75.2652435, -35.5135574, 60.2043304, -104.0989456, 110.7788010
1: -48.0108223, 63.5181313, -38.8392754, 51.8292198, -99.8400116, 102.3574066
2: -49.2012405, 63.6662178, -39.7609940, 51.6851311, -100.8863678, 103.4272079
3: -56.7119904, 73.5356216, -45.8723030, 60.1126862, -116.8246765, 119.4079285
4: -51.9156113, 73.6841812, -42.3089447, 59.8713531, -111.7869568, 115.9931259

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5540836, upper bound: 96.5802279
time: 0.91 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5530320, upper bound: 96.5563368
time: 1.02 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -45.3713455, 78.1234589, -35.5135574, 60.2043304, -105.5756760, 113.6370163
1: -49.6764984, 65.9766541, -38.8392754, 51.8292198, -101.5056839, 104.8159103
2: -50.8740158, 66.0815887, -39.7609940, 51.6851311, -102.5591354, 105.8425827
3: -58.8034401, 76.3701706, -45.8723030, 60.1126862, -118.9161224, 122.2424774
4: -53.7591057, 76.5252075, -42.3089447, 59.8713531, -113.6304321, 118.8341522

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 13

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5540836, upper bound: 96.5671023
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -96.5530320, upper bound: 96.5566725
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -43.8946152, 75.2652435, -34.2379265, 58.1512413, -102.0458527, 109.5031738
1: -48.0108223, 63.5181313, -37.4365273, 49.8464813, -97.8572769, 100.9546585
2: -49.2012405, 63.6662178, -38.3505554, 49.7083359, -98.9095764, 102.0167542
3: -56.7119904, 73.5356216, -44.2359123, 57.8597908, -114.5717773, 117.7715225
4: -51.9156113, 73.6841812, -40.8477364, 57.5693550, -109.4849396, 114.5319214

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 1

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -45.3713455, 78.1234589, -34.2379265, 58.1512413, -103.5225830, 112.3613739
1: -49.6764984, 65.9766541, -37.4365273, 49.8464813, -99.5229492, 103.4131622
2: -50.8740158, 66.0815887, -38.3505554, 49.7083359, -100.5823517, 104.4321365
3: -58.8034401, 76.3701706, -44.2359123, 57.8597908, -116.6632233, 120.6060715
4: -53.7591057, 76.5252075, -40.8477364, 57.5693550, -111.3284225, 117.3729401

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 3

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -44.4287033, 75.7598953, -35.5135574, 60.2043304, -104.6330338, 111.2734528
1: -48.5812683, 64.0594254, -38.8392754, 51.8292198, -100.4104843, 102.8986969
2: -49.7800446, 64.2237701, -39.7609940, 51.6851311, -101.4651794, 103.9847641
3: -57.3418617, 74.1730194, -45.8723030, 60.1126862, -117.4545441, 120.0453186
4: -52.4777603, 74.3543930, -42.3089447, 59.8713531, -112.3491058, 116.6633301

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.70 + 417.42 = 420.12 seconds
