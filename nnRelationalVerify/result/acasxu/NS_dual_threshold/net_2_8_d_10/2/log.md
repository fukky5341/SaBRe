## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_8.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 2)
Time budget: 420 seconds
Split limit: 100
Threshold: 0.019857408


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0103909, 0.0321280, 0.0103909, 0.0321280, -0.0217372, 0.0217372)
1: (-0.0221248, -0.0208590, -0.0221248, -0.0208590, -0.0012658, 0.0012658)
2: (0.0178454, 0.0198667, 0.0178454, 0.0198667, -0.0020212, 0.0020212)
3: (-0.0172454, -0.0152881, -0.0172454, -0.0152881, -0.0019573, 0.0019573)
4: (0.0197064, 0.0216206, 0.0197064, 0.0216206, -0.0019143, 0.0019143)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.81 + 0.77 = 3.58 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0206848, upper bound: 0.0206848

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0205398, upper bound: 0.0206737
time: 0.31 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206848, upper bound: 0.0206848
time: 0.32 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 0.87 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 0.87
Output dim: 0, lower bound: -0.0205398, upper bound: 0.0206737
NS_A2, status: Status.UNKNOWN, split count: 1, time: 0.87
Output dim: 0, lower bound: -0.0206848, upper bound: 0.0206848

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: 0.0159617, 0.0316399, 0.0103909, 0.0321280, -0.0161663, 0.0212491
1: -0.0219832, -0.0210485, -0.0221248, -0.0208590, -0.0011242, 0.0010763
2: 0.0186291, 0.0196666, 0.0178454, 0.0198667, -0.0012375, 0.0018211
3: -0.0171911, -0.0156491, -0.0172454, -0.0152881, -0.0019030, 0.0015963
4: 0.0197616, 0.0211237, 0.0197064, 0.0216206, -0.0018590, 0.0014174

Time for backsubstitution: 2.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0205296, upper bound: 0.0205296
time: 0.31 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0205296, upper bound: 0.0206737
time: 0.32 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: 0.0195159, 0.0227596, 0.0103909, 0.0321280, -0.0126121, 0.0123687
1: -0.0215229, -0.0212697, -0.0221248, -0.0208590, -0.0006639, 0.0008551
2: 0.0186858, 0.0191737, 0.0178454, 0.0198667, -0.0011808, 0.0013282
3: -0.0170585, -0.0163813, -0.0172454, -0.0152881, -0.0017704, 0.0008641
4: 0.0198993, 0.0204838, 0.0197064, 0.0216206, -0.0017213, 0.0007775

Time for backsubstitution: 2.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206737, upper bound: 0.0205311
time: 0.30 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206737, upper bound: 0.0206848
time: 0.30 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.41 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.41
Output dim: 0, lower bound: -0.0205296, upper bound: 0.0205296
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.41
Output dim: 0, lower bound: -0.0205296, upper bound: 0.0206737
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.41
Output dim: 0, lower bound: -0.0206737, upper bound: 0.0205311
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.41
Output dim: 0, lower bound: -0.0206737, upper bound: 0.0206848

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: 0.0159617, 0.0316399, 0.0159617, 0.0316399, -0.0156782, 0.0156782
1: -0.0219832, -0.0210485, -0.0219832, -0.0210485, -0.0009347, 0.0009347
2: 0.0186291, 0.0196666, 0.0186291, 0.0196666, -0.0010374, 0.0010374
3: -0.0171911, -0.0156491, -0.0171911, -0.0156491, -0.0015420, 0.0015420
4: 0.0197616, 0.0211237, 0.0197616, 0.0211237, -0.0013621, 0.0013621

Time for backsubstitution: 2.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0205139, upper bound: 0.0203518
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0203642, upper bound: 0.0203642
time: 0.31 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: 0.0159617, 0.0316399, 0.0195159, 0.0227596, -0.0067978, 0.0121241
1: -0.0219832, -0.0210485, -0.0215229, -0.0212697, -0.0007135, 0.0004744
2: 0.0186291, 0.0196666, 0.0186858, 0.0191737, -0.0005445, 0.0009807
3: -0.0171911, -0.0156491, -0.0170585, -0.0163813, -0.0008097, 0.0014094
4: 0.0197616, 0.0211237, 0.0198993, 0.0204838, -0.0007222, 0.0012244

Time for backsubstitution: 2.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0205139, upper bound: 0.0206435
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0203642, upper bound: 0.0206439
time: 0.31 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: 0.0195159, 0.0227596, 0.0159617, 0.0316399, -0.0121241, 0.0067978
1: -0.0215229, -0.0212697, -0.0219832, -0.0210485, -0.0004744, 0.0007135
2: 0.0186858, 0.0191737, 0.0186291, 0.0196666, -0.0009807, 0.0005445
3: -0.0170585, -0.0163813, -0.0171911, -0.0156491, -0.0014094, 0.0008097
4: 0.0198993, 0.0204838, 0.0197616, 0.0211237, -0.0012244, 0.0007222

Time for backsubstitution: 2.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206435, upper bound: 0.0205002
time: 0.32 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206439, upper bound: 0.0203657
time: 0.32 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: 0.0195159, 0.0227596, 0.0195159, 0.0227596, -0.0032437, 0.0032437
1: -0.0215229, -0.0212697, -0.0215229, -0.0212697, -0.0002532, 0.0002532
2: 0.0186858, 0.0191737, 0.0186858, 0.0191737, -0.0004878, 0.0004878
3: -0.0170585, -0.0163813, -0.0170585, -0.0163813, -0.0006772, 0.0006772
4: 0.0198993, 0.0204838, 0.0198993, 0.0204838, -0.0005845, 0.0005845

Time for backsubstitution: 2.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206569, upper bound: 0.0205125
time: 0.33 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206458, upper bound: 0.0205514
time: 0.32 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.47 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 0, lower bound: -0.0205139, upper bound: 0.0203518
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 0, lower bound: -0.0203642, upper bound: 0.0203642
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 0, lower bound: -0.0205139, upper bound: 0.0206435
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 0, lower bound: -0.0203642, upper bound: 0.0206439
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 0, lower bound: -0.0206435, upper bound: 0.0205002
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 0, lower bound: -0.0206439, upper bound: 0.0203657
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 0, lower bound: -0.0206569, upper bound: 0.0205125
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.47
Output dim: 0, lower bound: -0.0206458, upper bound: 0.0205514

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0162792, 0.0311817, 0.0159617, 0.0316399, -0.0153608, 0.0152200
1: -0.0219515, -0.0210579, -0.0219832, -0.0210485, -0.0009029, 0.0009253
2: 0.0186381, 0.0196478, 0.0186291, 0.0196666, -0.0010284, 0.0010186
3: -0.0171801, -0.0156917, -0.0171911, -0.0156491, -0.0015310, 0.0014993
4: 0.0197730, 0.0210815, 0.0197616, 0.0211237, -0.0013508, 0.0013199

Time for backsubstitution: 2.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0203518, upper bound: 0.0203518
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0203518, upper bound: 0.0203518
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0151246, 0.0300819, 0.0159617, 0.0316399, -0.0165153, 0.0141201
1: -0.0218969, -0.0210807, -0.0219832, -0.0210485, -0.0008484, 0.0009025
2: 0.0186162, 0.0196312, 0.0186291, 0.0196666, -0.0010503, 0.0010021
3: -0.0172251, -0.0157061, -0.0171911, -0.0156491, -0.0015759, 0.0014849
4: 0.0197272, 0.0210690, 0.0197616, 0.0211237, -0.0013965, 0.0013074

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0203518, upper bound: 0.0203642
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0203518, upper bound: 0.0203642
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0162792, 0.0311817, 0.0195159, 0.0227596, -0.0064804, 0.0116658
1: -0.0219515, -0.0210579, -0.0215229, -0.0212697, -0.0006818, 0.0004650
2: 0.0186381, 0.0196478, 0.0186858, 0.0191737, -0.0005355, 0.0009620
3: -0.0171801, -0.0156917, -0.0170585, -0.0163813, -0.0007988, 0.0013668
4: 0.0197730, 0.0210815, 0.0198993, 0.0204838, -0.0007109, 0.0011822

Time for backsubstitution: 2.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0203911, upper bound: 0.0206294
time: 0.32 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0204971, upper bound: 0.0206003
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0151246, 0.0300819, 0.0195159, 0.0227596, -0.0076349, 0.0105660
1: -0.0218969, -0.0210807, -0.0215229, -0.0212697, -0.0006272, 0.0004422
2: 0.0186162, 0.0196312, 0.0186858, 0.0191737, -0.0005574, 0.0009454
3: -0.0172251, -0.0157061, -0.0170585, -0.0163813, -0.0008437, 0.0013524
4: 0.0197272, 0.0210690, 0.0198993, 0.0204838, -0.0007566, 0.0011697

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0203518, upper bound: 0.0206315
time: 0.33 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0203518, upper bound: 0.0206439
time: 0.31 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: 0.0195159, 0.0227596, 0.0162792, 0.0311817, -0.0116658, 0.0064804
1: -0.0215229, -0.0212697, -0.0219515, -0.0210579, -0.0004650, 0.0006818
2: 0.0186858, 0.0191737, 0.0186381, 0.0196478, -0.0009620, 0.0005355
3: -0.0170585, -0.0163813, -0.0171801, -0.0156917, -0.0013668, 0.0007988
4: 0.0198993, 0.0204838, 0.0197730, 0.0210815, -0.0011822, 0.0007109

Time for backsubstitution: 2.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206294, upper bound: 0.0203911
time: 0.32 seconds

## Relational analysis of NS_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206003, upper bound: 0.0204971
time: 0.33 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: 0.0195159, 0.0227596, 0.0151246, 0.0300819, -0.0105660, 0.0076349
1: -0.0215229, -0.0212697, -0.0218969, -0.0210807, -0.0004422, 0.0006272
2: 0.0186858, 0.0191737, 0.0186162, 0.0196312, -0.0009454, 0.0005574
3: -0.0170585, -0.0163813, -0.0172251, -0.0157061, -0.0013524, 0.0008437
4: 0.0198993, 0.0204838, 0.0197272, 0.0210690, -0.0011697, 0.0007566

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206315, upper bound: 0.0203518
time: 0.33 seconds

## Relational analysis of NS_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206315, upper bound: 0.0203657
time: 0.33 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0208043, 0.0225723, 0.0201103, 0.0227122, -0.0019079, 0.0024621
1: -0.0214983, -0.0212676, -0.0215193, -0.0212801, -0.0002182, 0.0002517
2: 0.0187086, 0.0191650, 0.0186972, 0.0191613, -0.0004527, 0.0004678
3: -0.0170065, -0.0163937, -0.0170334, -0.0163970, -0.0006095, 0.0006397
4: 0.0199520, 0.0204754, 0.0199245, 0.0204709, -0.0005190, 0.0005508

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_A1

### Relational analysis result of NS_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206219, upper bound: 0.0199270
time: 0.33 seconds

## Relational analysis of NS_A2_B2_A1_A2

### Relational analysis result of NS_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206431, upper bound: 0.0204657
time: 0.33 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0196407, 0.0227297, 0.0195159, 0.0227596, -0.0031189, 0.0032138
1: -0.0215221, -0.0212729, -0.0215229, -0.0212697, -0.0002524, 0.0002500
2: 0.0186879, 0.0191708, 0.0186858, 0.0191737, -0.0004857, 0.0004850
3: -0.0170537, -0.0163849, -0.0170585, -0.0163813, -0.0006723, 0.0006736
4: 0.0199043, 0.0204809, 0.0198993, 0.0204838, -0.0005796, 0.0005816

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0205547, upper bound: 0.0204037
time: 0.33 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0205547, upper bound: 0.0205514
time: 0.32 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.50 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.50
Output dim: 0, lower bound: -0.0203518, upper bound: 0.0203518
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.50
Output dim: 0, lower bound: -0.0203518, upper bound: 0.0203518
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.50
Output dim: 0, lower bound: -0.0203518, upper bound: 0.0203642
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.50
Output dim: 0, lower bound: -0.0203518, upper bound: 0.0203642
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.50
Output dim: 0, lower bound: -0.0203911, upper bound: 0.0206294
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.50
Output dim: 0, lower bound: -0.0204971, upper bound: 0.0206003
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.50
Output dim: 0, lower bound: -0.0203518, upper bound: 0.0206315
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.50
Output dim: 0, lower bound: -0.0203518, upper bound: 0.0206439
NS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.50
Output dim: 0, lower bound: -0.0206294, upper bound: 0.0203911
NS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.50
Output dim: 0, lower bound: -0.0206003, upper bound: 0.0204971
NS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.50
Output dim: 0, lower bound: -0.0206315, upper bound: 0.0203518
NS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.50
Output dim: 0, lower bound: -0.0206315, upper bound: 0.0203657
NS_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 3.50
Output dim: 0, lower bound: -0.0206219, upper bound: 0.0199270
NS_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.50
Output dim: 0, lower bound: -0.0206431, upper bound: 0.0204657
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.50
Output dim: 0, lower bound: -0.0205547, upper bound: 0.0204037
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.50
Output dim: 0, lower bound: -0.0205547, upper bound: 0.0205514

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0162792, 0.0311817, 0.0162792, 0.0311817, -0.0149025, 0.0149025
1: -0.0219515, -0.0210579, -0.0219515, -0.0210579, -0.0008935, 0.0008935
2: 0.0186381, 0.0196478, 0.0186381, 0.0196478, -0.0010096, 0.0010096
3: -0.0171801, -0.0156917, -0.0171801, -0.0156917, -0.0014884, 0.0014884
4: 0.0197730, 0.0210815, 0.0197730, 0.0210815, -0.0013085, 0.0013085

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0162792, 0.0311817, 0.0151246, 0.0300819, -0.0138027, 0.0160571
1: -0.0219515, -0.0210579, -0.0218969, -0.0210807, -0.0008708, 0.0008390
2: 0.0186381, 0.0196478, 0.0186162, 0.0196312, -0.0009931, 0.0010315
3: -0.0171801, -0.0156917, -0.0172251, -0.0157061, -0.0014740, 0.0015333
4: 0.0197730, 0.0210815, 0.0197272, 0.0210690, -0.0012960, 0.0013543

Time for backsubstitution: 2.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 33

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0151246, 0.0300819, 0.0162792, 0.0311817, -0.0160571, 0.0138027
1: -0.0218969, -0.0210807, -0.0219515, -0.0210579, -0.0008390, 0.0008708
2: 0.0186162, 0.0196312, 0.0186381, 0.0196478, -0.0010315, 0.0009931
3: -0.0172251, -0.0157061, -0.0171801, -0.0156917, -0.0015333, 0.0014740
4: 0.0197272, 0.0210690, 0.0197730, 0.0210815, -0.0013543, 0.0012960

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0151246, 0.0300819, 0.0151246, 0.0300819, -0.0149572, 0.0149572
1: -0.0218969, -0.0210807, -0.0218969, -0.0210807, -0.0008162, 0.0008162
2: 0.0186162, 0.0196312, 0.0186162, 0.0196312, -0.0010150, 0.0010150
3: -0.0172251, -0.0157061, -0.0172251, -0.0157061, -0.0015189, 0.0015189
4: 0.0197272, 0.0210690, 0.0197272, 0.0210690, -0.0013418, 0.0013418

Time for backsubstitution: 2.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0168856, 0.0310751, 0.0208043, 0.0225723, -0.0056868, 0.0102708
1: -0.0219408, -0.0210716, -0.0214983, -0.0212676, -0.0006732, 0.0004267
2: 0.0186480, 0.0196156, 0.0187086, 0.0191650, -0.0005170, 0.0009070
3: -0.0171560, -0.0157544, -0.0170065, -0.0163937, -0.0007623, 0.0012521
4: 0.0197978, 0.0210292, 0.0199520, 0.0204754, -0.0006776, 0.0010772

Time for backsubstitution: 2.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0194169, upper bound: 0.0206284
time: 0.34 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0203010, upper bound: 0.0204001
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1_B1

### Relational analysis result of NS_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0198202, upper bound: 0.0206112
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A1_B1_B2

### Relational analysis result of NS_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0198202, upper bound: 0.0206294
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0162792, 0.0311817, 0.0196407, 0.0227297, -0.0064505, 0.0115410
1: -0.0219515, -0.0210579, -0.0215221, -0.0212729, -0.0006786, 0.0004642
2: 0.0186381, 0.0196478, 0.0186879, 0.0191708, -0.0005327, 0.0009598
3: -0.0171801, -0.0156917, -0.0170537, -0.0163849, -0.0007952, 0.0013619
4: 0.0197730, 0.0210815, 0.0199043, 0.0204809, -0.0007079, 0.0011772

Time for backsubstitution: 2.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B2_B1

### Relational analysis result of NS_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0204891, upper bound: 0.0206003
time: 0.33 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2

### Relational analysis result of NS_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0204891, upper bound: 0.0206003
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0151246, 0.0300819, 0.0197786, 0.0227177, -0.0075931, 0.0103033
1: -0.0218969, -0.0210807, -0.0215211, -0.0212834, -0.0006135, 0.0004405
2: 0.0186162, 0.0196312, 0.0186902, 0.0191599, -0.0005437, 0.0009410
3: -0.0172251, -0.0157061, -0.0170480, -0.0163988, -0.0008262, 0.0013419
4: 0.0197272, 0.0210690, 0.0199100, 0.0204693, -0.0007421, 0.0011590

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0192672, upper bound: 0.0206306
time: 0.32 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 13

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0151246, 0.0300819, 0.0184338, 0.0227259, -0.0076013, 0.0116481
1: -0.0218969, -0.0210807, -0.0215229, -0.0212756, -0.0006213, 0.0004422
2: 0.0186162, 0.0196312, 0.0186666, 0.0191629, -0.0005467, 0.0009646
3: -0.0172251, -0.0157061, -0.0171024, -0.0163941, -0.0008310, 0.0013963
4: 0.0197272, 0.0210690, 0.0198552, 0.0204728, -0.0007456, 0.0012138

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 13

## BFS NS instance: NS_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0208043, 0.0225723, 0.0168856, 0.0310751, -0.0102708, 0.0056868
1: -0.0214983, -0.0212676, -0.0219408, -0.0210716, -0.0004267, 0.0006732
2: 0.0187086, 0.0191650, 0.0186480, 0.0196156, -0.0009070, 0.0005170
3: -0.0170065, -0.0163937, -0.0171560, -0.0157544, -0.0012521, 0.0007623
4: 0.0199520, 0.0204754, 0.0197978, 0.0210292, -0.0010772, 0.0006776

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_B1_A1_B1

### Relational analysis result of NS_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206284, upper bound: 0.0194169
time: 0.32 seconds

## Relational analysis of NS_A2_B1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_B1_A1_B1

### Relational analysis result of NS_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0204001, upper bound: 0.0203010
time: 0.33 seconds

## Relational analysis of NS_A2_B1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_B1_A1_A1

### Relational analysis result of NS_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206112, upper bound: 0.0198202
time: 0.32 seconds

## Relational analysis of NS_A2_B1_B1_A1_A2

### Relational analysis result of NS_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206112, upper bound: 0.0203911
time: 0.32 seconds

## BFS NS instance: NS_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0196407, 0.0227297, 0.0162792, 0.0311817, -0.0115410, 0.0064505
1: -0.0215221, -0.0212729, -0.0219515, -0.0210579, -0.0004642, 0.0006786
2: 0.0186879, 0.0191708, 0.0186381, 0.0196478, -0.0009598, 0.0005327
3: -0.0170537, -0.0163849, -0.0171801, -0.0156917, -0.0013619, 0.0007952
4: 0.0199043, 0.0204809, 0.0197730, 0.0210815, -0.0011772, 0.0007079

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_B1_A2_A1

### Relational analysis result of NS_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206003, upper bound: 0.0204891
time: 0.32 seconds

## Relational analysis of NS_A2_B1_B1_A2_A2

### Relational analysis result of NS_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206003, upper bound: 0.0204971
time: 0.32 seconds

## BFS NS instance: NS_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0197786, 0.0227177, 0.0151246, 0.0300819, -0.0103033, 0.0075931
1: -0.0215211, -0.0212834, -0.0218969, -0.0210807, -0.0004405, 0.0006135
2: 0.0186902, 0.0191599, 0.0186162, 0.0196312, -0.0009410, 0.0005437
3: -0.0170480, -0.0163988, -0.0172251, -0.0157061, -0.0013419, 0.0008262
4: 0.0199100, 0.0204693, 0.0197272, 0.0210690, -0.0011590, 0.0007421

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_B2_A1_B1

### Relational analysis result of NS_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206244, upper bound: 0.0192672
time: 0.33 seconds

## Relational analysis of NS_A2_B1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0184338, 0.0227259, 0.0151246, 0.0300819, -0.0116481, 0.0076013
1: -0.0215229, -0.0212756, -0.0218969, -0.0210807, -0.0004422, 0.0006213
2: 0.0186666, 0.0191629, 0.0186162, 0.0196312, -0.0009646, 0.0005467
3: -0.0171024, -0.0163941, -0.0172251, -0.0157061, -0.0013963, 0.0008310
4: 0.0198552, 0.0204728, 0.0197272, 0.0210690, -0.0012138, 0.0007456

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A2_B2_A1_A1

### Backsubstitution after applying NS history:
0: 0.0209451, 0.0225421, 0.0201103, 0.0227122, -0.0017671, 0.0024318
1: -0.0214969, -0.0212741, -0.0215193, -0.0212801, -0.0002168, 0.0002452
2: 0.0187107, 0.0191514, 0.0186972, 0.0191613, -0.0004506, 0.0004542
3: -0.0170011, -0.0164111, -0.0170334, -0.0163970, -0.0006042, 0.0006223
4: 0.0199576, 0.0204610, 0.0199245, 0.0204709, -0.0005134, 0.0005364

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_A1_B1

### Relational analysis result of NS_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0205286, upper bound: 0.0199270
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A1_A1_B2

### Relational analysis result of NS_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0205286, upper bound: 0.0199270
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: 0.0210209, 0.0225073, 0.0201103, 0.0227122, -0.0016913, 0.0023970
1: -0.0214985, -0.0212876, -0.0215193, -0.0212801, -0.0002184, 0.0002317
2: 0.0187120, 0.0191457, 0.0186972, 0.0191613, -0.0004492, 0.0004486
3: -0.0169966, -0.0164179, -0.0170334, -0.0163970, -0.0005996, 0.0006155
4: 0.0199614, 0.0204549, 0.0199245, 0.0204709, -0.0005095, 0.0005304

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_A2_B1

### Relational analysis result of NS_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0205554, upper bound: 0.0204429
time: 0.33 seconds

## Relational analysis of NS_A2_B2_A1_A2_B2

### Relational analysis result of NS_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0205554, upper bound: 0.0204657
time: 0.33 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0196407, 0.0227297, 0.0208043, 0.0225723, -0.0029316, 0.0019254
1: -0.0215221, -0.0212729, -0.0214983, -0.0212676, -0.0002545, 0.0002254
2: 0.0186879, 0.0191708, 0.0187086, 0.0191650, -0.0004771, 0.0004622
3: -0.0170537, -0.0163849, -0.0170065, -0.0163937, -0.0006600, 0.0006216
4: 0.0199043, 0.0204809, 0.0199520, 0.0204754, -0.0005711, 0.0005289

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B1_B1

### Relational analysis result of NS_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199118, upper bound: 0.0202359
time: 0.33 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2

### Relational analysis result of NS_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0205040, upper bound: 0.0203660
time: 0.33 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0196407, 0.0227297, 0.0196407, 0.0227297, -0.0030890, 0.0030890
1: -0.0215221, -0.0212729, -0.0215221, -0.0212729, -0.0002492, 0.0002492
2: 0.0186879, 0.0191708, 0.0186879, 0.0191708, -0.0004829, 0.0004829
3: -0.0170537, -0.0163849, -0.0170537, -0.0163849, -0.0006688, 0.0006688
4: 0.0199043, 0.0204809, 0.0199043, 0.0204809, -0.0005766, 0.0005766

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0205213, upper bound: 0.0203943
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0205040, upper bound: 0.0203939
time: 0.33 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.55 seconds
NS_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 0, lower bound: -0.0198202, upper bound: 0.0206112
NS_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 0, lower bound: -0.0198202, upper bound: 0.0206294
NS_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 0, lower bound: -0.0204891, upper bound: 0.0206003
NS_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 0, lower bound: -0.0204891, upper bound: 0.0206003
NS_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 0, lower bound: -0.0206112, upper bound: 0.0198202
NS_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 0, lower bound: -0.0206112, upper bound: 0.0203911
NS_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 0, lower bound: -0.0206003, upper bound: 0.0204891
NS_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 0, lower bound: -0.0206003, upper bound: 0.0204971
NS_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 0, lower bound: -0.0205286, upper bound: 0.0199270
NS_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 0, lower bound: -0.0205286, upper bound: 0.0199270
NS_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 0, lower bound: -0.0205554, upper bound: 0.0204429
NS_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 0, lower bound: -0.0205554, upper bound: 0.0204657
NS_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 0, lower bound: -0.0199118, upper bound: 0.0202359
NS_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 0, lower bound: -0.0205040, upper bound: 0.0203660
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 0, lower bound: -0.0205213, upper bound: 0.0203943
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.55
Output dim: 0, lower bound: -0.0205040, upper bound: 0.0203939

## BFS NS instance: NS_A1_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: 0.0168856, 0.0310751, 0.0209451, 0.0225421, -0.0056565, 0.0101300
1: -0.0219408, -0.0210716, -0.0214969, -0.0212741, -0.0006667, 0.0004253
2: 0.0186480, 0.0196156, 0.0187107, 0.0191514, -0.0005033, 0.0009050
3: -0.0171560, -0.0157544, -0.0170011, -0.0164111, -0.0007449, 0.0012467
4: 0.0197978, 0.0210292, 0.0199576, 0.0204610, -0.0006632, 0.0010716

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

## BFS NS instance: NS_A1_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: 0.0168856, 0.0310751, 0.0210209, 0.0225073, -0.0056217, 0.0100542
1: -0.0219408, -0.0210716, -0.0214985, -0.0212876, -0.0006532, 0.0004269
2: 0.0186480, 0.0196156, 0.0187120, 0.0191457, -0.0004977, 0.0009036
3: -0.0171560, -0.0157544, -0.0169966, -0.0164179, -0.0007381, 0.0012422
4: 0.0197978, 0.0210292, 0.0199614, 0.0204549, -0.0006571, 0.0010678

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

## BFS NS instance: NS_A1_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: 0.0162792, 0.0311817, 0.0198755, 0.0226901, -0.0064109, 0.0113062
1: -0.0219515, -0.0210579, -0.0215207, -0.0212857, -0.0006658, 0.0004627
2: 0.0186381, 0.0196478, 0.0186918, 0.0191581, -0.0005200, 0.0009560
3: -0.0171801, -0.0156917, -0.0170443, -0.0164011, -0.0007790, 0.0013525
4: 0.0197730, 0.0210815, 0.0199138, 0.0204675, -0.0006945, 0.0011677

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 13

## BFS NS instance: NS_A1_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: 0.0162792, 0.0311817, 0.0185295, 0.0226950, -0.0064158, 0.0126522
1: -0.0219515, -0.0210579, -0.0215215, -0.0212783, -0.0006732, 0.0004636
2: 0.0186381, 0.0196478, 0.0186682, 0.0191596, -0.0005215, 0.0009796
3: -0.0171801, -0.0156917, -0.0170986, -0.0163983, -0.0007818, 0.0014068
4: 0.0197730, 0.0210815, 0.0198591, 0.0204693, -0.0006963, 0.0012224

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 13

## BFS NS instance: NS_A2_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: 0.0209451, 0.0225421, 0.0168856, 0.0310751, -0.0101300, 0.0056565
1: -0.0214969, -0.0212741, -0.0219408, -0.0210716, -0.0004253, 0.0006667
2: 0.0187107, 0.0191514, 0.0186480, 0.0196156, -0.0009050, 0.0005033
3: -0.0170011, -0.0164111, -0.0171560, -0.0157544, -0.0012467, 0.0007449
4: 0.0199576, 0.0204610, 0.0197978, 0.0210292, -0.0010716, 0.0006632

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 33

## BFS NS instance: NS_A2_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: 0.0210209, 0.0225073, 0.0168856, 0.0310751, -0.0100542, 0.0056217
1: -0.0214985, -0.0212876, -0.0219408, -0.0210716, -0.0004269, 0.0006532
2: 0.0187120, 0.0191457, 0.0186480, 0.0196156, -0.0009036, 0.0004977
3: -0.0169966, -0.0164179, -0.0171560, -0.0157544, -0.0012422, 0.0007381
4: 0.0199614, 0.0204549, 0.0197978, 0.0210292, -0.0010678, 0.0006571

Time for backsubstitution: 2.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 33

## BFS NS instance: NS_A2_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: 0.0198755, 0.0226901, 0.0162792, 0.0311817, -0.0113062, 0.0064109
1: -0.0215207, -0.0212857, -0.0219515, -0.0210579, -0.0004627, 0.0006658
2: 0.0186918, 0.0191581, 0.0186381, 0.0196478, -0.0009560, 0.0005200
3: -0.0170443, -0.0164011, -0.0171801, -0.0156917, -0.0013525, 0.0007790
4: 0.0199138, 0.0204675, 0.0197730, 0.0210815, -0.0011677, 0.0006945

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A2_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: 0.0185295, 0.0226950, 0.0162792, 0.0311817, -0.0126522, 0.0064158
1: -0.0215215, -0.0212783, -0.0219515, -0.0210579, -0.0004636, 0.0006732
2: 0.0186682, 0.0191596, 0.0186381, 0.0196478, -0.0009796, 0.0005215
3: -0.0170986, -0.0163983, -0.0171801, -0.0156917, -0.0014068, 0.0007818
4: 0.0198591, 0.0204693, 0.0197730, 0.0210815, -0.0012224, 0.0006963

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A2_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0209451, 0.0225421, 0.0208043, 0.0225723, -0.0016273, 0.0017378
1: -0.0214969, -0.0212741, -0.0214983, -0.0212676, -0.0002293, 0.0002242
2: 0.0187107, 0.0191514, 0.0187086, 0.0191650, -0.0004543, 0.0004427
3: -0.0170011, -0.0164111, -0.0170065, -0.0163937, -0.0006074, 0.0005954
4: 0.0199576, 0.0204610, 0.0199520, 0.0204754, -0.0005178, 0.0005090

Time for backsubstitution: 2.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_A1_B1_B1

### Relational analysis result of NS_A2_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199516, upper bound: 0.0199270
time: 0.33 seconds

## Relational analysis of NS_A2_B2_A1_A1_B1_B2

### Relational analysis result of NS_A2_B2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199516, upper bound: 0.0199270
time: 0.32 seconds

## BFS NS instance: NS_A2_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0209451, 0.0225421, 0.0196407, 0.0227297, -0.0017846, 0.0029014
1: -0.0214969, -0.0212741, -0.0215221, -0.0212729, -0.0002240, 0.0002480
2: 0.0187107, 0.0191514, 0.0186879, 0.0191708, -0.0004602, 0.0004634
3: -0.0170011, -0.0164111, -0.0170537, -0.0163849, -0.0006162, 0.0006426
4: 0.0199576, 0.0204610, 0.0199043, 0.0204809, -0.0005233, 0.0005567

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_A1_B2_B1

### Relational analysis result of NS_A2_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0205281, upper bound: 0.0195780
time: 0.32 seconds

## Relational analysis of NS_A2_B2_A1_A1_B2_B2

### Relational analysis result of NS_A2_B2_A1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0194546, upper bound: 0.0195743
time: 0.31 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0210209, 0.0225073, 0.0208043, 0.0225723, -0.0015514, 0.0017030
1: -0.0214985, -0.0212876, -0.0214983, -0.0212676, -0.0002309, 0.0002107
2: 0.0187120, 0.0191457, 0.0187086, 0.0191650, -0.0004530, 0.0004371
3: -0.0169966, -0.0164179, -0.0170065, -0.0163937, -0.0006029, 0.0005886
4: 0.0199614, 0.0204549, 0.0199520, 0.0204754, -0.0005140, 0.0005030

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_A2_B1_B1

### Relational analysis result of NS_A2_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199783, upper bound: 0.0204283
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A1_A2_B1_B2

### Relational analysis result of NS_A2_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199783, upper bound: 0.0204429
time: 0.33 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0210209, 0.0225073, 0.0196407, 0.0227297, -0.0017088, 0.0028666
1: -0.0214985, -0.0212876, -0.0215221, -0.0212729, -0.0002256, 0.0002345
2: 0.0187120, 0.0191457, 0.0186879, 0.0191708, -0.0004588, 0.0004578
3: -0.0169966, -0.0164179, -0.0170537, -0.0163849, -0.0006117, 0.0006358
4: 0.0199614, 0.0204549, 0.0199043, 0.0204809, -0.0005195, 0.0005506

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_A2_B2_B1

### Relational analysis result of NS_A2_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0205548, upper bound: 0.0195298
time: 0.33 seconds

## Relational analysis of NS_A2_B2_A1_A2_B2_B2

### Relational analysis result of NS_A2_B2_A1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0194813, upper bound: 0.0195239
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: 0.0196407, 0.0227297, 0.0209451, 0.0225421, -0.0029014, 0.0017846
1: -0.0215221, -0.0212729, -0.0214969, -0.0212741, -0.0002480, 0.0002240
2: 0.0186879, 0.0191708, 0.0187107, 0.0191514, -0.0004634, 0.0004602
3: -0.0170537, -0.0163849, -0.0170011, -0.0164111, -0.0006426, 0.0006162
4: 0.0199043, 0.0204809, 0.0199576, 0.0204610, -0.0005567, 0.0005233

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0195828, upper bound: 0.0202306
time: 0.35 seconds

## Relational analysis of NS_A2_B2_A2_B1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0195769, upper bound: 0.0193907
time: 0.32 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: 0.0196407, 0.0227297, 0.0210209, 0.0225073, -0.0028666, 0.0017088
1: -0.0215221, -0.0212729, -0.0214985, -0.0212876, -0.0002345, 0.0002256
2: 0.0186879, 0.0191708, 0.0187120, 0.0191457, -0.0004578, 0.0004588
3: -0.0170537, -0.0163849, -0.0169966, -0.0164179, -0.0006358, 0.0006117
4: 0.0199043, 0.0204809, 0.0199614, 0.0204549, -0.0005506, 0.0005195

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0195052, upper bound: 0.0202675
time: 0.33 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0194648, upper bound: 0.0194189
time: 0.33 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0198755, 0.0226901, 0.0196407, 0.0227297, -0.0028542, 0.0030493
1: -0.0215207, -0.0212857, -0.0215221, -0.0212729, -0.0002478, 0.0002364
2: 0.0186918, 0.0191581, 0.0186879, 0.0191708, -0.0004790, 0.0004702
3: -0.0170443, -0.0164011, -0.0170537, -0.0163849, -0.0006594, 0.0006526
4: 0.0199138, 0.0204675, 0.0199043, 0.0204809, -0.0005671, 0.0005632

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0203989, upper bound: 0.0203906
time: 0.35 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0203989, upper bound: 0.0203939
time: 0.35 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0185295, 0.0226950, 0.0196407, 0.0227297, -0.0042002, 0.0030543
1: -0.0215215, -0.0212783, -0.0215221, -0.0212729, -0.0002486, 0.0002438
2: 0.0186682, 0.0191596, 0.0186879, 0.0191708, -0.0005026, 0.0004717
3: -0.0170986, -0.0163983, -0.0170537, -0.0163849, -0.0007137, 0.0006554
4: 0.0198591, 0.0204693, 0.0199043, 0.0204809, -0.0006218, 0.0005650

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B2_A2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0194300, upper bound: 0.0202764
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0194241, upper bound: 0.0194189
time: 0.34 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.10 seconds
NS_A2_B2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -0.0199516, upper bound: 0.0199270
NS_A2_B2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -0.0199516, upper bound: 0.0199270
NS_A2_B2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -0.0205281, upper bound: 0.0195780
NS_A2_B2_A1_A1_B2_B2, status: Status.VERIFIED, split count: 6, time: 4.10
Output dim: 0, lower bound: -0.0194546, upper bound: 0.0195743
NS_A2_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -0.0199783, upper bound: 0.0204283
NS_A2_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -0.0199783, upper bound: 0.0204429
NS_A2_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -0.0205548, upper bound: 0.0195298
NS_A2_B2_A1_A2_B2_B2, status: Status.VERIFIED, split count: 6, time: 4.10
Output dim: 0, lower bound: -0.0194813, upper bound: 0.0195239
NS_A2_B2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -0.0195828, upper bound: 0.0202306
NS_A2_B2_A2_B1_B1_A2, status: Status.VERIFIED, split count: 6, time: 4.10
Output dim: 0, lower bound: -0.0195769, upper bound: 0.0193907
NS_A2_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -0.0195052, upper bound: 0.0202675
NS_A2_B2_A2_B1_B2_A2, status: Status.VERIFIED, split count: 6, time: 4.10
Output dim: 0, lower bound: -0.0194648, upper bound: 0.0194189
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -0.0203989, upper bound: 0.0203906
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -0.0203989, upper bound: 0.0203939
NS_A2_B2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -0.0194300, upper bound: 0.0202764
NS_A2_B2_A2_B2_A2_A2, status: Status.VERIFIED, split count: 6, time: 4.10
Output dim: 0, lower bound: -0.0194241, upper bound: 0.0194189

## BFS NS instance: NS_A2_B2_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: 0.0209451, 0.0225421, 0.0209451, 0.0225421, -0.0015970, 0.0015970
1: -0.0214969, -0.0212741, -0.0214969, -0.0212741, -0.0002227, 0.0002227
2: 0.0187107, 0.0191514, 0.0187107, 0.0191514, -0.0004407, 0.0004407
3: -0.0170011, -0.0164111, -0.0170011, -0.0164111, -0.0005901, 0.0005901
4: 0.0199576, 0.0204610, 0.0199576, 0.0204610, -0.0005034, 0.0005034

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_A1_B1_B1_A1

### Relational analysis result of NS_A2_B2_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0195544, upper bound: 0.0199176
time: 0.32 seconds

## Relational analysis of NS_A2_B2_A1_A1_B1_B1_A2

### Relational analysis result of NS_A2_B2_A1_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0195544, upper bound: 0.0195725
time: 0.32 seconds

## BFS NS instance: NS_A2_B2_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: 0.0209451, 0.0225421, 0.0210209, 0.0225073, -0.0015622, 0.0015212
1: -0.0214969, -0.0212741, -0.0214985, -0.0212876, -0.0002093, 0.0002244
2: 0.0187107, 0.0191514, 0.0187120, 0.0191457, -0.0004351, 0.0004393
3: -0.0170011, -0.0164111, -0.0169966, -0.0164179, -0.0005833, 0.0005856
4: 0.0199576, 0.0204610, 0.0199614, 0.0204549, -0.0004973, 0.0004996

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_A1_B1_B2_B1

### Relational analysis result of NS_A2_B2_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199433, upper bound: 0.0195780
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A1_A1_B1_B2_B2

### Relational analysis result of NS_A2_B2_A1_A1_B1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0195544, upper bound: 0.0195743
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: 0.0209451, 0.0225421, 0.0197344, 0.0227257, -0.0017807, 0.0028077
1: -0.0214969, -0.0212741, -0.0215221, -0.0212739, -0.0002230, 0.0002479
2: 0.0187107, 0.0191514, 0.0186896, 0.0191696, -0.0004589, 0.0004618
3: -0.0170011, -0.0164111, -0.0170499, -0.0163865, -0.0006146, 0.0006389
4: 0.0199576, 0.0204610, 0.0199081, 0.0204796, -0.0005220, 0.0005529

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_A1_B2_B1_B1

### Relational analysis result of NS_A2_B2_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206102, upper bound: 0.0195773
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A1_A1_B2_B1_B2

### Relational analysis result of NS_A2_B2_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206102, upper bound: 0.0195780
time: 0.33 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: 0.0210209, 0.0225073, 0.0209451, 0.0225421, -0.0015212, 0.0015622
1: -0.0214985, -0.0212876, -0.0214969, -0.0212741, -0.0002244, 0.0002093
2: 0.0187120, 0.0191457, 0.0187107, 0.0191514, -0.0004393, 0.0004351
3: -0.0169966, -0.0164179, -0.0170011, -0.0164111, -0.0005856, 0.0005833
4: 0.0199614, 0.0204549, 0.0199576, 0.0204610, -0.0004996, 0.0004973

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_A2_B1_B1_A1

### Relational analysis result of NS_A2_B2_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0195811, upper bound: 0.0204269
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A1_A2_B1_B1_A2

### Relational analysis result of NS_A2_B2_A1_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0195811, upper bound: 0.0195299
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: 0.0210209, 0.0225073, 0.0210209, 0.0225073, -0.0014864, 0.0014864
1: -0.0214985, -0.0212876, -0.0214985, -0.0212876, -0.0002109, 0.0002109
2: 0.0187120, 0.0191457, 0.0187120, 0.0191457, -0.0004337, 0.0004337
3: -0.0169966, -0.0164179, -0.0169966, -0.0164179, -0.0005787, 0.0005787
4: 0.0199614, 0.0204549, 0.0199614, 0.0204549, -0.0004935, 0.0004935

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_A2_B1_B2_A1

### Relational analysis result of NS_A2_B2_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0195811, upper bound: 0.0204332
time: 0.33 seconds

## Relational analysis of NS_A2_B2_A1_A2_B1_B2_A2

### Relational analysis result of NS_A2_B2_A1_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0195811, upper bound: 0.0195299
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: 0.0210209, 0.0225073, 0.0197344, 0.0227257, -0.0017048, 0.0027729
1: -0.0214985, -0.0212876, -0.0215221, -0.0212739, -0.0002246, 0.0002344
2: 0.0187120, 0.0191457, 0.0186896, 0.0191696, -0.0004575, 0.0004562
3: -0.0169966, -0.0164179, -0.0170499, -0.0163865, -0.0006101, 0.0006321
4: 0.0199614, 0.0204549, 0.0199081, 0.0204796, -0.0005182, 0.0005468

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_A2_B2_B1_A1

### Relational analysis result of NS_A2_B2_A1_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0194813, upper bound: 0.0195239
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A1_A2_B2_B1_A2

### Relational analysis result of NS_A2_B2_A1_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0194813, upper bound: 0.0195239
time: 0.33 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0197344, 0.0227257, 0.0209451, 0.0225421, -0.0028077, 0.0017807
1: -0.0215221, -0.0212739, -0.0214969, -0.0212741, -0.0002479, 0.0002230
2: 0.0186896, 0.0191696, 0.0187107, 0.0191514, -0.0004618, 0.0004589
3: -0.0170499, -0.0163865, -0.0170011, -0.0164111, -0.0006389, 0.0006146
4: 0.0199081, 0.0204796, 0.0199576, 0.0204610, -0.0005529, 0.0005220

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B1_B1_A1_A1

### Relational analysis result of NS_A2_B2_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0195828, upper bound: 0.0200917
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A2_B1_B1_A1_A2

### Relational analysis result of NS_A2_B2_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0195828, upper bound: 0.0202306
time: 0.35 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0197344, 0.0227257, 0.0210209, 0.0225073, -0.0027729, 0.0017048
1: -0.0215221, -0.0212739, -0.0214985, -0.0212876, -0.0002344, 0.0002246
2: 0.0186896, 0.0191696, 0.0187120, 0.0191457, -0.0004562, 0.0004575
3: -0.0170499, -0.0163865, -0.0169966, -0.0164179, -0.0006321, 0.0006101
4: 0.0199081, 0.0204796, 0.0199614, 0.0204549, -0.0005468, 0.0005182

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0194648, upper bound: 0.0194189
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0194648, upper bound: 0.0194189
time: 0.35 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0198755, 0.0226901, 0.0198755, 0.0226901, -0.0028146, 0.0028146
1: -0.0215207, -0.0212857, -0.0215207, -0.0212857, -0.0002350, 0.0002350
2: 0.0186918, 0.0191581, 0.0186918, 0.0191581, -0.0004663, 0.0004663
3: -0.0170443, -0.0164011, -0.0170443, -0.0164011, -0.0006432, 0.0006432
4: 0.0199138, 0.0204675, 0.0199138, 0.0204675, -0.0005537, 0.0005537

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0194473, upper bound: 0.0201924
time: 0.35 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0194473, upper bound: 0.0196157
time: 0.33 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0198755, 0.0226901, 0.0185295, 0.0226950, -0.0028195, 0.0041606
1: -0.0215207, -0.0212857, -0.0215215, -0.0212783, -0.0002424, 0.0002359
2: 0.0186918, 0.0191581, 0.0186682, 0.0191596, -0.0004678, 0.0004899
3: -0.0170443, -0.0164011, -0.0170986, -0.0163983, -0.0006460, 0.0006975
4: 0.0199138, 0.0204675, 0.0198591, 0.0204693, -0.0005555, 0.0006084

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0204960, upper bound: 0.0196321
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0194473, upper bound: 0.0196157
time: 0.37 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_A1

### Backsubstitution after applying NS history:
0: 0.0186350, 0.0226922, 0.0196407, 0.0227297, -0.0040947, 0.0030515
1: -0.0215215, -0.0212791, -0.0215221, -0.0212729, -0.0002486, 0.0002430
2: 0.0186700, 0.0191581, 0.0186879, 0.0191708, -0.0005008, 0.0004702
3: -0.0170943, -0.0164003, -0.0170537, -0.0163849, -0.0007094, 0.0006534
4: 0.0198634, 0.0204677, 0.0199043, 0.0204809, -0.0006175, 0.0005634

Time for backsubstitution: 2.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B2_A2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0194300, upper bound: 0.0202621
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0194300, upper bound: 0.0202764
time: 0.34 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 3.60 seconds
NS_A2_B2_A1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -0.0195544, upper bound: 0.0199176
NS_A2_B2_A1_A1_B1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.60
Output dim: 0, lower bound: -0.0195544, upper bound: 0.0195725
NS_A2_B2_A1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -0.0199433, upper bound: 0.0195780
NS_A2_B2_A1_A1_B1_B2_B2, status: Status.VERIFIED, split count: 7, time: 3.60
Output dim: 0, lower bound: -0.0195544, upper bound: 0.0195743
NS_A2_B2_A1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -0.0206102, upper bound: 0.0195773
NS_A2_B2_A1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -0.0206102, upper bound: 0.0195780
NS_A2_B2_A1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -0.0195811, upper bound: 0.0204269
NS_A2_B2_A1_A2_B1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.60
Output dim: 0, lower bound: -0.0195811, upper bound: 0.0195299
NS_A2_B2_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -0.0195811, upper bound: 0.0204332
NS_A2_B2_A1_A2_B1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.60
Output dim: 0, lower bound: -0.0195811, upper bound: 0.0195299
NS_A2_B2_A1_A2_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.60
Output dim: 0, lower bound: -0.0194813, upper bound: 0.0195239
NS_A2_B2_A1_A2_B2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.60
Output dim: 0, lower bound: -0.0194813, upper bound: 0.0195239
NS_A2_B2_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -0.0195828, upper bound: 0.0200917
NS_A2_B2_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -0.0195828, upper bound: 0.0202306
NS_A2_B2_A2_B1_B2_A1_B1, status: Status.VERIFIED, split count: 7, time: 3.60
Output dim: 0, lower bound: -0.0194648, upper bound: 0.0194189
NS_A2_B2_A2_B1_B2_A1_B2, status: Status.VERIFIED, split count: 7, time: 3.60
Output dim: 0, lower bound: -0.0194648, upper bound: 0.0194189
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -0.0194473, upper bound: 0.0201924
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.60
Output dim: 0, lower bound: -0.0194473, upper bound: 0.0196157
NS_A2_B2_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -0.0204960, upper bound: 0.0196321
NS_A2_B2_A2_B2_A1_B2_B2, status: Status.VERIFIED, split count: 7, time: 3.60
Output dim: 0, lower bound: -0.0194473, upper bound: 0.0196157
NS_A2_B2_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -0.0194300, upper bound: 0.0202621
NS_A2_B2_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.60
Output dim: 0, lower bound: -0.0194300, upper bound: 0.0202764

## BFS NS instance: NS_A2_B2_A1_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0209469, 0.0225406, 0.0209451, 0.0225421, -0.0015952, 0.0015955
1: -0.0214969, -0.0212742, -0.0214969, -0.0212741, -0.0002227, 0.0002227
2: 0.0187107, 0.0191504, 0.0187107, 0.0191514, -0.0004407, 0.0004397
3: -0.0170011, -0.0164123, -0.0170011, -0.0164111, -0.0005900, 0.0005888
4: 0.0199577, 0.0204600, 0.0199576, 0.0204610, -0.0005033, 0.0005024

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_A1_B1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_A1_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0196074, upper bound: 0.0195725
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A1_A1_B1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_A1_B1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0196074, upper bound: 0.0195725
time: 0.33 seconds

## BFS NS instance: NS_A2_B2_A1_A1_B1_B2_B1

### Backsubstitution after applying NS history:
0: 0.0209451, 0.0225421, 0.0210242, 0.0225057, -0.0015606, 0.0015179
1: -0.0214969, -0.0212741, -0.0214985, -0.0212878, -0.0002091, 0.0002243
2: 0.0187107, 0.0191514, 0.0187121, 0.0191445, -0.0004338, 0.0004393
3: -0.0170011, -0.0164111, -0.0169965, -0.0164195, -0.0005817, 0.0005855
4: 0.0199576, 0.0204610, 0.0199615, 0.0204536, -0.0004960, 0.0004995

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_A1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_A1_B1_B2_B1_A1

### Relational analysis result of NS_A2_B2_A1_A1_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0195544, upper bound: 0.0195743
time: 0.35 seconds

## Relational analysis of NS_A2_B2_A1_A1_B1_B2_B1_A2

### Relational analysis result of NS_A2_B2_A1_A1_B1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0195544, upper bound: 0.0195743
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A1_A1_B2_B1_B1

### Backsubstitution after applying NS history:
0: 0.0209451, 0.0225421, 0.0199707, 0.0226861, -0.0017410, 0.0025714
1: -0.0214969, -0.0212741, -0.0215206, -0.0212867, -0.0002102, 0.0002465
2: 0.0187107, 0.0191514, 0.0186935, 0.0191568, -0.0004461, 0.0004579
3: -0.0170011, -0.0164111, -0.0170405, -0.0164028, -0.0005983, 0.0006294
4: 0.0199576, 0.0204610, 0.0199177, 0.0204661, -0.0005085, 0.0005433

Time for backsubstitution: 2.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_A1_B2_B1_B1_A1

### Relational analysis result of NS_A2_B2_A1_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206102, upper bound: 0.0195773
time: 0.35 seconds

## Relational analysis of NS_A2_B2_A1_A1_B2_B1_B1_A2

### Relational analysis result of NS_A2_B2_A1_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206102, upper bound: 0.0195773
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A1_A1_B2_B1_B2

### Backsubstitution after applying NS history:
0: 0.0209451, 0.0225421, 0.0186350, 0.0226922, -0.0017471, 0.0039071
1: -0.0214969, -0.0212741, -0.0215215, -0.0212791, -0.0002178, 0.0002473
2: 0.0187107, 0.0191514, 0.0186700, 0.0191581, -0.0004475, 0.0004814
3: -0.0170011, -0.0164111, -0.0170943, -0.0164003, -0.0006009, 0.0006832
4: 0.0199576, 0.0204610, 0.0198634, 0.0204677, -0.0005101, 0.0005976

Time for backsubstitution: 2.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_A1_B2_B1_B2_A1

### Relational analysis result of NS_A2_B2_A1_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206102, upper bound: 0.0195780
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A1_A1_B2_B1_B2_A2

### Relational analysis result of NS_A2_B2_A1_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206102, upper bound: 0.0195780
time: 0.33 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0210242, 0.0225057, 0.0209451, 0.0225421, -0.0015179, 0.0015606
1: -0.0214985, -0.0212878, -0.0214969, -0.0212741, -0.0002243, 0.0002091
2: 0.0187121, 0.0191445, 0.0187107, 0.0191514, -0.0004393, 0.0004338
3: -0.0169965, -0.0164195, -0.0170011, -0.0164111, -0.0005855, 0.0005817
4: 0.0199615, 0.0204536, 0.0199576, 0.0204610, -0.0004995, 0.0004960

Time for backsubstitution: 2.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_A2_B1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_A2_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0196341, upper bound: 0.0195343
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A1_A2_B1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_A2_B1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0196341, upper bound: 0.0195343
time: 0.35 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0210242, 0.0225057, 0.0210209, 0.0225073, -0.0014831, 0.0014848
1: -0.0214985, -0.0212878, -0.0214985, -0.0212876, -0.0002109, 0.0002107
2: 0.0187121, 0.0191445, 0.0187120, 0.0191457, -0.0004337, 0.0004324
3: -0.0169965, -0.0164195, -0.0169966, -0.0164179, -0.0005786, 0.0005771
4: 0.0199615, 0.0204536, 0.0199614, 0.0204549, -0.0004934, 0.0004922

Time for backsubstitution: 2.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_A2_B1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_A2_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0195811, upper bound: 0.0195299
time: 0.35 seconds

## Relational analysis of NS_A2_B2_A1_A2_B1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_A2_B1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0195811, upper bound: 0.0195299
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: 0.0199707, 0.0226861, 0.0209451, 0.0225421, -0.0025714, 0.0017410
1: -0.0215206, -0.0212867, -0.0214969, -0.0212741, -0.0002465, 0.0002102
2: 0.0186935, 0.0191568, 0.0187107, 0.0191514, -0.0004579, 0.0004461
3: -0.0170405, -0.0164028, -0.0170011, -0.0164111, -0.0006294, 0.0005983
4: 0.0199177, 0.0204661, 0.0199576, 0.0204610, -0.0005433, 0.0005085

Time for backsubstitution: 2.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1_B1_A1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0195828, upper bound: 0.0200917
time: 0.35 seconds

## Relational analysis of NS_A2_B2_A2_B1_B1_A1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0195828, upper bound: 0.0200917
time: 0.32 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: 0.0186350, 0.0226922, 0.0209451, 0.0225421, -0.0039071, 0.0017471
1: -0.0215215, -0.0212791, -0.0214969, -0.0212741, -0.0002473, 0.0002178
2: 0.0186700, 0.0191581, 0.0187107, 0.0191514, -0.0004814, 0.0004475
3: -0.0170943, -0.0164003, -0.0170011, -0.0164111, -0.0006832, 0.0006009
4: 0.0198634, 0.0204677, 0.0199576, 0.0204610, -0.0005976, 0.0005101

Time for backsubstitution: 2.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1_B1_A1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0195828, upper bound: 0.0202306
time: 0.35 seconds

## Relational analysis of NS_A2_B2_A2_B1_B1_A1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0195828, upper bound: 0.0202306
time: 0.35 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0199707, 0.0226861, 0.0198755, 0.0226901, -0.0027194, 0.0028106
1: -0.0215206, -0.0212867, -0.0215207, -0.0212857, -0.0002349, 0.0002339
2: 0.0186935, 0.0191568, 0.0186918, 0.0191581, -0.0004647, 0.0004650
3: -0.0170405, -0.0164028, -0.0170443, -0.0164011, -0.0006394, 0.0006414
4: 0.0199177, 0.0204661, 0.0199138, 0.0204675, -0.0005498, 0.0005523

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0196599, upper bound: 0.0196204
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0196599, upper bound: 0.0196204
time: 0.35 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: 0.0198755, 0.0226901, 0.0186350, 0.0226922, -0.0028167, 0.0040550
1: -0.0215207, -0.0212857, -0.0215215, -0.0212791, -0.0002415, 0.0002358
2: 0.0186918, 0.0191581, 0.0186700, 0.0191581, -0.0004663, 0.0004881
3: -0.0170443, -0.0164011, -0.0170943, -0.0164003, -0.0006440, 0.0006932
4: 0.0199138, 0.0204675, 0.0198634, 0.0204677, -0.0005539, 0.0006041

Time for backsubstitution: 2.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0194473, upper bound: 0.0196157
time: 0.35 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0194473, upper bound: 0.0196157
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0186350, 0.0226922, 0.0198755, 0.0226901, -0.0040550, 0.0028167
1: -0.0215215, -0.0212791, -0.0215207, -0.0212857, -0.0002358, 0.0002415
2: 0.0186700, 0.0191581, 0.0186918, 0.0191581, -0.0004881, 0.0004663
3: -0.0170943, -0.0164003, -0.0170443, -0.0164011, -0.0006932, 0.0006440
4: 0.0198634, 0.0204677, 0.0199138, 0.0204675, -0.0006041, 0.0005539

Time for backsubstitution: 2.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B2_A2_A1_B1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0194300, upper bound: 0.0202621
time: 0.35 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_A1_B1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0194300, upper bound: 0.0202621
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0186350, 0.0226922, 0.0185295, 0.0226950, -0.0040599, 0.0041627
1: -0.0215215, -0.0212791, -0.0215215, -0.0212783, -0.0002432, 0.0002424
2: 0.0186700, 0.0191581, 0.0186682, 0.0191596, -0.0004896, 0.0004899
3: -0.0170943, -0.0164003, -0.0170986, -0.0163983, -0.0006960, 0.0006983
4: 0.0198634, 0.0204677, 0.0198591, 0.0204693, -0.0006059, 0.0006086

Time for backsubstitution: 2.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B2_A2_A1_B2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0194300, upper bound: 0.0202764
time: 0.35 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_A1_B2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0194300, upper bound: 0.0202764
time: 0.34 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 3.63 seconds
NS_A2_B2_A1_A1_B1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.63
Output dim: 0, lower bound: -0.0196074, upper bound: 0.0195725
NS_A2_B2_A1_A1_B1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.63
Output dim: 0, lower bound: -0.0196074, upper bound: 0.0195725
NS_A2_B2_A1_A1_B1_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.63
Output dim: 0, lower bound: -0.0195544, upper bound: 0.0195743
NS_A2_B2_A1_A1_B1_B2_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.63
Output dim: 0, lower bound: -0.0195544, upper bound: 0.0195743
NS_A2_B2_A1_A1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 0, lower bound: -0.0206102, upper bound: 0.0195773
NS_A2_B2_A1_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 0, lower bound: -0.0206102, upper bound: 0.0195773
NS_A2_B2_A1_A1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 0, lower bound: -0.0206102, upper bound: 0.0195780
NS_A2_B2_A1_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 0, lower bound: -0.0206102, upper bound: 0.0195780
NS_A2_B2_A1_A2_B1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.63
Output dim: 0, lower bound: -0.0196341, upper bound: 0.0195343
NS_A2_B2_A1_A2_B1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.63
Output dim: 0, lower bound: -0.0196341, upper bound: 0.0195343
NS_A2_B2_A1_A2_B1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.63
Output dim: 0, lower bound: -0.0195811, upper bound: 0.0195299
NS_A2_B2_A1_A2_B1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.63
Output dim: 0, lower bound: -0.0195811, upper bound: 0.0195299
NS_A2_B2_A2_B1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 0, lower bound: -0.0195828, upper bound: 0.0200917
NS_A2_B2_A2_B1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 0, lower bound: -0.0195828, upper bound: 0.0200917
NS_A2_B2_A2_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 0, lower bound: -0.0195828, upper bound: 0.0202306
NS_A2_B2_A2_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 0, lower bound: -0.0195828, upper bound: 0.0202306
NS_A2_B2_A2_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.63
Output dim: 0, lower bound: -0.0196599, upper bound: 0.0196204
NS_A2_B2_A2_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.63
Output dim: 0, lower bound: -0.0196599, upper bound: 0.0196204
NS_A2_B2_A2_B2_A1_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.63
Output dim: 0, lower bound: -0.0194473, upper bound: 0.0196157
NS_A2_B2_A2_B2_A1_B2_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.63
Output dim: 0, lower bound: -0.0194473, upper bound: 0.0196157
NS_A2_B2_A2_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 0, lower bound: -0.0194300, upper bound: 0.0202621
NS_A2_B2_A2_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 0, lower bound: -0.0194300, upper bound: 0.0202621
NS_A2_B2_A2_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 0, lower bound: -0.0194300, upper bound: 0.0202764
NS_A2_B2_A2_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 0, lower bound: -0.0194300, upper bound: 0.0202764

## BFS NS instance: NS_A2_B2_A1_A1_B2_B1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0209469, 0.0225406, 0.0199707, 0.0226861, -0.0017392, 0.0025699
1: -0.0214969, -0.0212742, -0.0215206, -0.0212867, -0.0002101, 0.0002464
2: 0.0187107, 0.0191504, 0.0186935, 0.0191568, -0.0004461, 0.0004569
3: -0.0170011, -0.0164123, -0.0170405, -0.0164028, -0.0005982, 0.0006282
4: 0.0199577, 0.0204600, 0.0199177, 0.0204661, -0.0005084, 0.0005423

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A2_B2_A1_A1_B2_B1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0208929, 0.0227255, 0.0199707, 0.0226861, -0.0017933, 0.0027548
1: -0.0215001, -0.0212701, -0.0215206, -0.0212867, -0.0002134, 0.0002505
2: 0.0187082, 0.0191588, 0.0186935, 0.0191568, -0.0004486, 0.0004653
3: -0.0170047, -0.0164017, -0.0170405, -0.0164028, -0.0006019, 0.0006388
4: 0.0199544, 0.0204688, 0.0199177, 0.0204661, -0.0005116, 0.0005511

Time for backsubstitution: 2.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A2_B2_A1_A1_B2_B1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0209469, 0.0225406, 0.0186350, 0.0226922, -0.0017453, 0.0039055
1: -0.0214969, -0.0212742, -0.0215215, -0.0212791, -0.0002177, 0.0002472
2: 0.0187107, 0.0191504, 0.0186700, 0.0191581, -0.0004474, 0.0004804
3: -0.0170011, -0.0164123, -0.0170943, -0.0164003, -0.0006008, 0.0006820
4: 0.0199577, 0.0204600, 0.0198634, 0.0204677, -0.0005100, 0.0005966

Time for backsubstitution: 2.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_A1_B2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A2_B2_A1_A1_B2_B1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0208929, 0.0227255, 0.0186350, 0.0226922, -0.0017993, 0.0040904
1: -0.0215001, -0.0212701, -0.0215215, -0.0212791, -0.0002210, 0.0002514
2: 0.0187082, 0.0191588, 0.0186700, 0.0191581, -0.0004500, 0.0004888
3: -0.0170047, -0.0164017, -0.0170943, -0.0164003, -0.0006044, 0.0006926
4: 0.0199544, 0.0204688, 0.0198634, 0.0204677, -0.0005132, 0.0006054

Time for backsubstitution: 2.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_A1_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A2_B2_A2_B1_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0199707, 0.0226861, 0.0209469, 0.0225406, -0.0025699, 0.0017392
1: -0.0215206, -0.0212867, -0.0214969, -0.0212742, -0.0002464, 0.0002101
2: 0.0186935, 0.0191568, 0.0187107, 0.0191504, -0.0004569, 0.0004461
3: -0.0170405, -0.0164028, -0.0170011, -0.0164123, -0.0006282, 0.0005982
4: 0.0199177, 0.0204661, 0.0199577, 0.0204600, -0.0005423, 0.0005084

Time for backsubstitution: 2.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

## BFS NS instance: NS_A2_B2_A2_B1_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0199707, 0.0226861, 0.0208929, 0.0227255, -0.0027548, 0.0017933
1: -0.0215206, -0.0212867, -0.0215001, -0.0212701, -0.0002505, 0.0002134
2: 0.0186935, 0.0191568, 0.0187082, 0.0191588, -0.0004653, 0.0004486
3: -0.0170405, -0.0164028, -0.0170047, -0.0164017, -0.0006388, 0.0006019
4: 0.0199177, 0.0204661, 0.0199544, 0.0204688, -0.0005511, 0.0005116

Time for backsubstitution: 2.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

## BFS NS instance: NS_A2_B2_A2_B1_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0186350, 0.0226922, 0.0209469, 0.0225406, -0.0039055, 0.0017453
1: -0.0215215, -0.0212791, -0.0214969, -0.0212742, -0.0002472, 0.0002177
2: 0.0186700, 0.0191581, 0.0187107, 0.0191504, -0.0004804, 0.0004474
3: -0.0170943, -0.0164003, -0.0170011, -0.0164123, -0.0006820, 0.0006008
4: 0.0198634, 0.0204677, 0.0199577, 0.0204600, -0.0005966, 0.0005100

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_B1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## BFS NS instance: NS_A2_B2_A2_B1_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0186350, 0.0226922, 0.0208929, 0.0227255, -0.0040904, 0.0017993
1: -0.0215215, -0.0212791, -0.0215001, -0.0212701, -0.0002514, 0.0002210
2: 0.0186700, 0.0191581, 0.0187082, 0.0191588, -0.0004888, 0.0004500
3: -0.0170943, -0.0164003, -0.0170047, -0.0164017, -0.0006926, 0.0006044
4: 0.0198634, 0.0204677, 0.0199544, 0.0204688, -0.0006054, 0.0005132

Time for backsubstitution: 2.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_B1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## BFS NS instance: NS_A2_B2_A2_B2_A2_A1_B1_B1

### Backsubstitution after applying NS history:
0: 0.0186350, 0.0226922, 0.0199707, 0.0226861, -0.0040511, 0.0027215
1: -0.0215215, -0.0212791, -0.0215206, -0.0212867, -0.0002348, 0.0002415
2: 0.0186700, 0.0191581, 0.0186935, 0.0191568, -0.0004868, 0.0004647
3: -0.0170943, -0.0164003, -0.0170405, -0.0164028, -0.0006915, 0.0006402
4: 0.0198634, 0.0204677, 0.0199177, 0.0204661, -0.0006027, 0.0005500

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A2_B2_A2_B2_A2_A1_B1_B2

### Backsubstitution after applying NS history:
0: 0.0186350, 0.0226922, 0.0200177, 0.0228201, -0.0041851, 0.0026745
1: -0.0215215, -0.0212791, -0.0215232, -0.0212781, -0.0002434, 0.0002440
2: 0.0186700, 0.0191581, 0.0186953, 0.0191831, -0.0005131, 0.0004629
3: -0.0170943, -0.0164003, -0.0170375, -0.0163691, -0.0007252, 0.0006372
4: 0.0198634, 0.0204677, 0.0199204, 0.0204939, -0.0006305, 0.0005473

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A2_B2_A2_B2_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: 0.0186350, 0.0226922, 0.0186350, 0.0226922, -0.0040572, 0.0040572
1: -0.0215215, -0.0212791, -0.0215215, -0.0212791, -0.0002423, 0.0002423
2: 0.0186700, 0.0191581, 0.0186700, 0.0191581, -0.0004881, 0.0004881
3: -0.0170943, -0.0164003, -0.0170943, -0.0164003, -0.0006940, 0.0006940
4: 0.0198634, 0.0204677, 0.0198634, 0.0204677, -0.0006043, 0.0006043

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

## BFS NS instance: NS_A2_B2_A2_B2_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: 0.0186350, 0.0226922, 0.0191664, 0.0228172, -0.0041822, 0.0035258
1: -0.0215215, -0.0212791, -0.0215203, -0.0212736, -0.0002479, 0.0002411
2: 0.0186700, 0.0191581, 0.0186795, 0.0191780, -0.0005081, 0.0004787
3: -0.0170943, -0.0164003, -0.0170729, -0.0163752, -0.0007191, 0.0006726
4: 0.0198634, 0.0204677, 0.0198849, 0.0204884, -0.0006250, 0.0005828

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.58 + 293.27 = 296.85 seconds
