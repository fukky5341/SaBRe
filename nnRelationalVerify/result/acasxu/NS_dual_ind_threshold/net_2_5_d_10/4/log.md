## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_5.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 4)
Time budget: 420 seconds
Split limit: 100
Threshold: 71.14967792064


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-86.8210449, 164.3859558, -86.8210449, 164.3859558, -251.2070007, 251.2070007)
1: (-29.7035065, 57.3727303, -29.7035065, 57.3727303, -87.0762329, 87.0762329)
2: (-15.8138723, 59.4475250, -15.8138723, 59.4475250, -75.2613983, 75.2613983)
3: (-33.6637955, 71.3761292, -33.6637955, 71.3761292, -105.0399246, 105.0399246)
4: (-20.2065468, 58.7098618, -20.2065468, 58.7098618, -78.9164124, 78.9164124)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.19 + 1.92 = 4.11 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -71.1567936, upper bound: 71.1567936

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1537061, upper bound: 71.1498773
time: 0.71 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1505939, upper bound: 71.1505939
time: 0.77 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.65 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.65
Output dim: 4, lower bound: -71.1537061, upper bound: 71.1498773
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.65
Output dim: 4, lower bound: -71.1505939, upper bound: 71.1505939

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -71.3806305, 133.5665894, -86.8210449, 164.3859558, -235.7665863, 220.3876343
1: -24.0552578, 46.1637917, -29.7035065, 57.3727303, -81.4279861, 75.8672943
2: -12.8562984, 47.8482742, -15.8138723, 59.4475250, -72.3038177, 63.6621475
3: -27.3332100, 57.3620796, -33.6637955, 71.3761292, -98.7093353, 91.0258789
4: -16.4428463, 47.2423553, -20.2065468, 58.7098618, -75.1527023, 67.4488983

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1497563, upper bound: 71.1497563
time: 0.68 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1497563, upper bound: 71.1497563
time: 0.70 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -173.6494904, 328.2228088, -84.0670090, 159.1587219, -332.8082275, 412.2897949
1: -59.6009102, 114.0641785, -28.7131901, 55.4299622, -115.0308685, 142.7773743
2: -31.6865673, 116.7040787, -15.2946281, 57.4346123, -89.1211624, 131.9987030
3: -67.0983734, 142.6331329, -32.5652199, 68.9846878, -136.0830536, 175.1983490
4: -40.7940636, 115.2602386, -19.5454044, 56.7068024, -97.5008698, 134.8056488

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1497563, upper bound: 71.1505939
time: 0.71 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1497563, upper bound: 71.1505939
time: 0.73 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.62 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.62
Output dim: 4, lower bound: -71.1497563, upper bound: 71.1497563
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.62
Output dim: 4, lower bound: -71.1497563, upper bound: 71.1497563
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.62
Output dim: 4, lower bound: -71.1497563, upper bound: 71.1505939
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.62
Output dim: 4, lower bound: -71.1497563, upper bound: 71.1505939

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -71.3806305, 133.5665894, -71.3806305, 133.5665894, -204.9472198, 204.9472198
1: -24.0552578, 46.1637917, -24.0552578, 46.1637917, -70.2190399, 70.2190399
2: -12.8562984, 47.8482742, -12.8562984, 47.8482742, -60.7045746, 60.7045746
3: -27.3332100, 57.3620796, -27.3332100, 57.3620796, -84.6952896, 84.6952896
4: -16.4428463, 47.2423553, -16.4428463, 47.2423553, -63.6851921, 63.6851845

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1525013, upper bound: 71.1460324
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1525013, upper bound: 71.1460324
time: 0.70 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -71.3806305, 133.5665894, -173.6494904, 328.2228088, -399.6034546, 307.2160645
1: -24.0552578, 46.1637917, -59.6009102, 114.0641785, -138.1194305, 105.7647018
2: -12.8562984, 47.8482742, -31.6865673, 116.7040787, -129.5603790, 79.5348206
3: -27.3332100, 57.3620796, -67.0983734, 142.6331329, -169.9663391, 124.4604492
4: -16.4428463, 47.2423553, -40.7940636, 115.2602386, -131.7030640, 88.0363998

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1525013, upper bound: 71.1460324
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1525469, upper bound: 71.1486700
time: 0.72 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -173.6494904, 328.2228088, -71.3806305, 133.5665894, -307.2160645, 399.6033936
1: -59.6009102, 114.0641785, -24.0552578, 46.1637917, -105.7647018, 138.1194305
2: -31.6865673, 116.7040787, -12.8562984, 47.8482742, -79.5348206, 129.5603638
3: -67.0983734, 142.6331329, -27.3332100, 57.3620796, -124.4604492, 169.9663391
4: -40.7940636, 115.2602386, -16.4428463, 47.2423553, -88.0363998, 131.7030792

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1461208, upper bound: 71.1503937
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1461175, upper bound: 71.1500696
time: 0.68 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -173.6494904, 328.2228088, -173.6494904, 328.2228088, -501.8723145, 501.8722839
1: -59.6009102, 114.0641785, -59.6009102, 114.0641785, -173.6650848, 173.6650848
2: -31.6865673, 116.7040787, -31.6865673, 116.7040787, -148.3906250, 148.3906250
3: -67.0983734, 142.6331329, -67.0983734, 142.6331329, -209.7315063, 209.7315063
4: -40.7940636, 115.2602386, -40.7940636, 115.2602386, -156.0543060, 156.0543060

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1461208, upper bound: 71.1503937
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1461175, upper bound: 71.1500696
time: 0.71 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.59 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.59
Output dim: 4, lower bound: -71.1525013, upper bound: 71.1460324
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.59
Output dim: 4, lower bound: -71.1525013, upper bound: 71.1460324
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.59
Output dim: 4, lower bound: -71.1525013, upper bound: 71.1460324
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.59
Output dim: 4, lower bound: -71.1525469, upper bound: 71.1486700
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.59
Output dim: 4, lower bound: -71.1461208, upper bound: 71.1503937
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.59
Output dim: 4, lower bound: -71.1461175, upper bound: 71.1500696
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.59
Output dim: 4, lower bound: -71.1461208, upper bound: 71.1503937
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.59
Output dim: 4, lower bound: -71.1461175, upper bound: 71.1500696

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -61.7794189, 115.1032562, -71.3806305, 133.5665894, -195.3460083, 186.4838867
1: -20.7804565, 39.8721085, -24.0552578, 46.1637917, -66.9442444, 63.9273491
2: -11.0813818, 41.2324982, -12.8562984, 47.8482742, -58.9296570, 54.0887947
3: -23.6343994, 49.4206123, -27.3332100, 57.3620796, -80.9964752, 76.7538223
4: -14.1975718, 40.5734253, -16.4428463, 47.2423553, -61.4399261, 57.0162659

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1508633, upper bound: 71.1508633
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1508633, upper bound: 71.1508633
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -76.9331360, 143.8970490, -71.3806305, 133.5665894, -210.4997253, 215.2776794
1: -25.9769173, 49.5727768, -24.0552578, 46.1637917, -72.1407089, 73.6280289
2: -13.8924885, 51.3647881, -12.8562984, 47.8482742, -61.7407608, 64.2210693
3: -29.4370537, 61.6523323, -27.3332100, 57.3620796, -86.7991333, 88.9855347
4: -17.7362938, 50.7621155, -16.4428463, 47.2423553, -64.9786377, 67.2049561

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1510876, upper bound: 71.1532186
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1510876, upper bound: 71.1534185
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -61.7794189, 115.1032562, -173.6494904, 328.2228088, -390.0021973, 288.7527466
1: -20.7804565, 39.8721085, -59.6009102, 114.0641785, -134.8446198, 99.4730225
2: -11.0813818, 41.2324982, -31.6865673, 116.7040787, -127.7854614, 72.9190445
3: -23.6343994, 49.4206123, -67.0983734, 142.6331329, -166.2675171, 116.5189819
4: -14.1975718, 40.5734253, -40.7940636, 115.2602386, -129.4578094, 81.3674774

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1475437, upper bound: 71.1434959
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1474721, upper bound: 71.1386643
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -76.9331360, 143.8970490, -173.6494904, 328.2228088, -405.1559448, 317.5465088
1: -25.9769173, 49.5727768, -59.6009102, 114.0641785, -140.0410919, 109.1736908
2: -13.8924885, 51.3647881, -31.6865673, 116.7040787, -130.5965729, 83.0513382
3: -29.4370537, 61.6523323, -67.0983734, 142.6331329, -172.0701904, 128.7506866
4: -17.7362938, 50.7621155, -40.7940636, 115.2602386, -132.9965210, 91.5561829

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1501311, upper bound: 71.1468263
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1511300, upper bound: 71.1486700
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1511300, upper bound: 71.1486700
time: 0.73 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -171.4372559, 323.7672119, -71.3806305, 133.5665894, -305.0038452, 395.1478271
1: -58.7259674, 112.2340546, -24.0552578, 46.1637917, -104.8897400, 136.2893066
2: -31.2524872, 114.8404465, -12.8562984, 47.8482742, -79.1007309, 127.6967316
3: -66.1143951, 140.3671417, -27.3332100, 57.3620796, -123.4764709, 167.7003479
4: -40.2243309, 113.4743729, -16.4428463, 47.2423553, -87.4666748, 129.9172211

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1434959, upper bound: 71.1482855
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1461175, upper bound: 71.1504007
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1461175, upper bound: 71.1504007
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -187.3775482, 356.1024170, -70.7322769, 132.2265320, -319.6039429, 426.8346558
1: -64.4656143, 122.4750595, -23.8197651, 45.7157402, -110.1813354, 146.2947845
2: -34.3348808, 125.0910263, -12.7352104, 47.3742332, -81.7091141, 137.8262329
3: -72.4417419, 153.2488251, -27.0745106, 56.8022003, -129.2439423, 180.3233337
4: -44.1374283, 123.8815765, -16.2882938, 46.7772560, -90.9146729, 140.1698456

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1386643, upper bound: 71.1474721
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1461175, upper bound: 71.1504007
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1461175, upper bound: 71.1504007
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -171.4372559, 323.7672119, -173.6494904, 328.2228088, -499.6600647, 497.4166565
1: -58.7259674, 112.2340546, -59.6009102, 114.0641785, -172.7901306, 171.8349609
2: -31.2524872, 114.8404465, -31.6865673, 116.7040787, -147.9565277, 146.5270081
3: -66.1143951, 140.3671417, -67.0983734, 142.6331329, -208.7475281, 207.4655151
4: -40.2243309, 113.4743729, -40.7940636, 115.2602386, -155.4845734, 154.2684326

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1461175, upper bound: 71.1500696
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1461175, upper bound: 71.1461175
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -187.3775482, 356.1024170, -173.1243744, 327.1806946, -514.5582275, 529.2266846
1: -64.4656143, 122.4750595, -59.4033241, 113.6562576, -178.1218109, 181.8783417
2: -34.3348808, 125.0910263, -31.5857677, 116.2849731, -150.6198425, 156.6767883
3: -72.4417419, 153.2488251, -66.8736954, 142.1254883, -214.5672302, 220.1225128
4: -44.1374283, 123.8815765, -40.6620789, 114.8539124, -158.9913330, 164.5436249

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1461175, upper bound: 71.1500696
time: 0.92 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1478313, upper bound: 71.1500696
time: 0.74 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.88 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 4, lower bound: -71.1508633, upper bound: 71.1508633
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 4, lower bound: -71.1508633, upper bound: 71.1508633
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 4, lower bound: -71.1510876, upper bound: 71.1532186
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 4, lower bound: -71.1510876, upper bound: 71.1534185
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.88
Output dim: 4, lower bound: -71.1475437, upper bound: 71.1434959
NS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 3.88
Output dim: 4, lower bound: -71.1474721, upper bound: 71.1386643
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 4, lower bound: -71.1511300, upper bound: 71.1486700
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 4, lower bound: -71.1511300, upper bound: 71.1486700
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 4, lower bound: -71.1461175, upper bound: 71.1504007
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 4, lower bound: -71.1461175, upper bound: 71.1504007
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 4, lower bound: -71.1461175, upper bound: 71.1504007
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 4, lower bound: -71.1461175, upper bound: 71.1504007
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 4, lower bound: -71.1461175, upper bound: 71.1500696
NS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 3.88
Output dim: 4, lower bound: -71.1461175, upper bound: 71.1461175
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 4, lower bound: -71.1461175, upper bound: 71.1500696
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 4, lower bound: -71.1478313, upper bound: 71.1500696

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -61.7794189, 115.1032562, -61.7794189, 115.1032562, -176.8826752, 176.8826752
1: -20.7804565, 39.8721085, -20.7804565, 39.8721085, -60.6525650, 60.6525650
2: -11.0813818, 41.2324982, -11.0813818, 41.2324982, -52.3138771, 52.3138771
3: -23.6343994, 49.4206123, -23.6343994, 49.4206123, -73.0550079, 73.0550079
4: -14.1975718, 40.5734253, -14.1975718, 40.5734253, -54.7709961, 54.7709961

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1484468, upper bound: 71.1471023
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1480506, upper bound: 71.1476450
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -61.7794189, 115.1032562, -76.9331360, 143.8970490, -205.6764679, 192.0363770
1: -20.7804565, 39.8721085, -25.9769173, 49.5727768, -70.3532333, 65.8490143
2: -11.0813818, 41.2324982, -13.8924885, 51.3647881, -62.4461594, 55.1249847
3: -23.6343994, 49.4206123, -29.4370537, 61.6523323, -85.2867279, 78.8576660
4: -14.1975718, 40.5734253, -17.7362938, 50.7621155, -64.9596863, 58.3097153

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1484468, upper bound: 71.1471023
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1480506, upper bound: 71.1476450
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -76.9331360, 143.8970490, -61.7794189, 115.1032562, -192.0363770, 205.6764679
1: -25.9769173, 49.5727768, -20.7804565, 39.8721085, -65.8490143, 70.3532333
2: -13.8924885, 51.3647881, -11.0813818, 41.2324982, -55.1249847, 62.4461708
3: -29.4370537, 61.6523323, -23.6343994, 49.4206123, -78.8576660, 85.2867279
4: -17.7362938, 50.7621155, -14.1975718, 40.5734253, -58.3097153, 64.9596863

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1484468, upper bound: 71.1490414
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1476450, upper bound: 71.1489000
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -76.9331360, 143.8970490, -76.9331360, 143.8970490, -220.8301697, 220.8301697
1: -25.9769173, 49.5727768, -25.9769173, 49.5727768, -75.5496979, 75.5496979
2: -13.8924885, 51.3647881, -13.8924885, 51.3647881, -65.2572784, 65.2572784
3: -29.4370537, 61.6523323, -29.4370537, 61.6523323, -91.0893784, 91.0893784
4: -17.7362938, 50.7621155, -17.7362938, 50.7621155, -68.4984131, 68.4984131

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1485552, upper bound: 71.1490414
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1476450, upper bound: 71.1489000
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -76.9331360, 143.8970490, -163.4310760, 308.0657654, -384.9989014, 307.3280334
1: -25.9769173, 49.5727768, -55.8503914, 106.4486694, -132.4255829, 105.4231720
2: -13.8924885, 51.3647881, -29.7153702, 108.9095840, -122.8020706, 81.0801392
3: -29.4370537, 61.6523323, -62.7612305, 133.0799561, -162.5170135, 124.4135590
4: -17.7362938, 50.7621155, -38.2336426, 107.4938660, -125.2301559, 88.9957504

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1480294, upper bound: 71.1453397
time: 0.90 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1492445, upper bound: 71.1451984
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -76.9331360, 143.8970490, -177.1676636, 334.6231689, -411.5563049, 321.0646667
1: -25.9769173, 49.5727768, -60.8054695, 116.1694260, -142.1463470, 110.3782425
2: -13.8924885, 51.3647881, -32.3098526, 118.8604507, -132.7529449, 83.6746368
3: -29.4370537, 61.6523323, -68.4273071, 145.2766571, -174.7137146, 130.0796356
4: -17.7362938, 50.7621155, -41.5744553, 117.3710938, -135.1073456, 92.3365555

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1480294, upper bound: 71.1453397
time: 0.83 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1476450, upper bound: 71.1451984
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -171.4372559, 323.7672119, -69.4512100, 129.9380951, -301.3753662, 393.2184143
1: -58.7259674, 112.2340546, -23.3971233, 44.9333267, -103.6592636, 135.6311798
2: -31.2524872, 114.8404465, -12.4948254, 46.5807381, -77.8332062, 127.3352509
3: -66.1143951, 140.3671417, -26.5884361, 55.8011513, -121.9155350, 166.9555511
4: -40.2243309, 113.4743729, -15.9917002, 45.9513550, -86.1756897, 129.4660645

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1457307, upper bound: 71.1479698
time: 0.78 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1455128, upper bound: 71.1495720
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -171.4254303, 323.7448730, -82.1872330, 153.8225403, -325.2478638, 405.9320679
1: -58.7217751, 112.2259979, -27.7218342, 53.0266228, -111.7483902, 139.9478149
2: -31.2502956, 114.8322449, -14.8963575, 54.7776871, -86.0279694, 129.7285919
3: -66.1098099, 140.3570862, -31.5214996, 66.0433502, -132.1531525, 171.8785706
4: -40.2215080, 113.4663696, -19.0578575, 54.3203735, -94.5418854, 132.5242310

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1457307, upper bound: 71.1479698
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1455128, upper bound: 71.1495720
time: 0.73 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -187.3775482, 356.1024170, -69.4512100, 129.9380951, -317.3155518, 425.5536194
1: -64.4656143, 122.4750595, -23.3971233, 44.9333267, -109.3989029, 145.8721619
2: -34.3348808, 125.0910263, -12.4948254, 46.5807381, -80.9156189, 137.5858459
3: -72.4417419, 153.2488251, -26.5884361, 55.8011513, -128.2428589, 179.8372498
4: -44.1374283, 123.8815765, -15.9917002, 45.9513550, -90.0887833, 139.8732605

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1457709, upper bound: 71.1482783
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1448711, upper bound: 71.1490506
time: 1.05 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -187.3775482, 356.1024170, -82.1864853, 153.8213501, -341.1988220, 438.2889099
1: -64.4656143, 122.4750595, -27.7215652, 53.0260353, -117.4916229, 150.1965790
2: -34.3348808, 125.0910263, -14.8962116, 54.7770805, -89.1119614, 139.9872437
3: -72.4417419, 153.2488251, -31.5211735, 66.0426865, -138.4844055, 184.7700043
4: -44.1374283, 123.8815765, -19.0576878, 54.3197784, -98.4571915, 142.9392700

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1457709, upper bound: 71.1482783
time: 0.80 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1448711, upper bound: 71.1490506
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -171.4372559, 323.7672119, -171.4372559, 323.7672119, -495.2044678, 495.2044678
1: -58.7259674, 112.2340546, -58.7259674, 112.2340546, -170.9600220, 170.9600220
2: -31.2524872, 114.8404465, -31.2524872, 114.8404465, -146.0929108, 146.0929108
3: -66.1143951, 140.3671417, -66.1143951, 140.3671417, -206.4815369, 206.4815369
4: -40.2243309, 113.4743729, -40.2243309, 113.4743729, -153.6986847, 153.6986847

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1424016, upper bound: 71.1423984
time: 0.88 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1423955, upper bound: 71.1433949
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -187.3775482, 356.1024170, -171.4372559, 323.7672119, -511.1447449, 527.5396729
1: -64.4656143, 122.4750595, -58.7259674, 112.2340546, -176.6996613, 181.2010193
2: -34.3348808, 125.0910263, -31.2524872, 114.8404465, -149.1753235, 156.3434906
3: -72.4417419, 153.2488251, -66.1143951, 140.3671417, -212.8088531, 219.3632202
4: -44.1374283, 123.8815765, -40.2243309, 113.4743729, -157.6118011, 164.1058960

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1440286, upper bound: 71.1423980
time: 0.94 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1423956, upper bound: 71.1424258
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -187.3775482, 356.1024170, -187.3775482, 356.1024170, -543.4799194, 543.4799194
1: -64.4656143, 122.4750595, -64.4656143, 122.4750595, -186.9406586, 186.9406586
2: -34.3348808, 125.0910263, -34.3348808, 125.0910263, -159.4259033, 159.4259033
3: -72.4417419, 153.2488251, -72.4417419, 153.2488251, -225.6905670, 225.6905670
4: -44.1374283, 123.8815765, -44.1374283, 123.8815765, -168.0189819, 168.0189819

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1440286, upper bound: 71.1423981
time: 0.80 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1423956, upper bound: 71.1424258
time: 1.02 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 7.11 seconds
NS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 7.11
Output dim: 4, lower bound: -71.1484468, upper bound: 71.1471023
NS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 7.11
Output dim: 4, lower bound: -71.1480506, upper bound: 71.1476450
NS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 7.11
Output dim: 4, lower bound: -71.1484468, upper bound: 71.1471023
NS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 7.11
Output dim: 4, lower bound: -71.1480506, upper bound: 71.1476450
NS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 7.11
Output dim: 4, lower bound: -71.1484468, upper bound: 71.1490414
NS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 7.11
Output dim: 4, lower bound: -71.1476450, upper bound: 71.1489000
NS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 7.11
Output dim: 4, lower bound: -71.1485552, upper bound: 71.1490414
NS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 7.11
Output dim: 4, lower bound: -71.1476450, upper bound: 71.1489000
NS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 7.11
Output dim: 4, lower bound: -71.1480294, upper bound: 71.1453397
NS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 7.11
Output dim: 4, lower bound: -71.1492445, upper bound: 71.1451984
NS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 7.11
Output dim: 4, lower bound: -71.1480294, upper bound: 71.1453397
NS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 7.11
Output dim: 4, lower bound: -71.1476450, upper bound: 71.1451984
NS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 7.11
Output dim: 4, lower bound: -71.1457307, upper bound: 71.1479698
NS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 7.11
Output dim: 4, lower bound: -71.1455128, upper bound: 71.1495720
NS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 7.11
Output dim: 4, lower bound: -71.1457307, upper bound: 71.1479698
NS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 7.11
Output dim: 4, lower bound: -71.1455128, upper bound: 71.1495720
NS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 7.11
Output dim: 4, lower bound: -71.1457709, upper bound: 71.1482783
NS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 7.11
Output dim: 4, lower bound: -71.1448711, upper bound: 71.1490506
NS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 7.11
Output dim: 4, lower bound: -71.1457709, upper bound: 71.1482783
NS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 7.11
Output dim: 4, lower bound: -71.1448711, upper bound: 71.1490506
NS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 7.11
Output dim: 4, lower bound: -71.1424016, upper bound: 71.1423984
NS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 7.11
Output dim: 4, lower bound: -71.1423955, upper bound: 71.1433949
NS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 7.11
Output dim: 4, lower bound: -71.1440286, upper bound: 71.1423980
NS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 7.11
Output dim: 4, lower bound: -71.1423956, upper bound: 71.1424258
NS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 7.11
Output dim: 4, lower bound: -71.1440286, upper bound: 71.1423981
NS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 7.11
Output dim: 4, lower bound: -71.1423956, upper bound: 71.1424258

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 4.11 + 128.43 = 132.54 seconds
