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
execution time: IAR + RelationalAnalysis = 2.85 + 0.76 = 3.61 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0206848, upper bound: 0.0206848

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33

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
- Time for NS candidates: 0.86 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 0.86
Output dim: 0, lower bound: -0.0205398, upper bound: 0.0206737
NS_A2, status: Status.UNKNOWN, split count: 1, time: 0.86
Output dim: 0, lower bound: -0.0206848, upper bound: 0.0206848

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: 0.0159617, 0.0316399, 0.0103909, 0.0321280, -0.0161663, 0.0212491
1: -0.0219832, -0.0210485, -0.0221248, -0.0208590, -0.0011242, 0.0010763
2: 0.0186291, 0.0196666, 0.0178454, 0.0198667, -0.0012375, 0.0018211
3: -0.0171911, -0.0156491, -0.0172454, -0.0152881, -0.0019030, 0.0015963
4: 0.0197616, 0.0211237, 0.0197064, 0.0216206, -0.0018590, 0.0014174

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.20 seconds

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

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206737, upper bound: 0.0205311
time: 0.31 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206737, upper bound: 0.0206848
time: 0.31 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.45 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.45
Output dim: 0, lower bound: -0.0205296, upper bound: 0.0205296
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.45
Output dim: 0, lower bound: -0.0205296, upper bound: 0.0206737
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.45
Output dim: 0, lower bound: -0.0206737, upper bound: 0.0205311
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.45
Output dim: 0, lower bound: -0.0206737, upper bound: 0.0206848

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: 0.0159617, 0.0316399, 0.0159617, 0.0316399, -0.0156782, 0.0156782
1: -0.0219832, -0.0210485, -0.0219832, -0.0210485, -0.0009347, 0.0009347
2: 0.0186291, 0.0196666, 0.0186291, 0.0196666, -0.0010374, 0.0010374
3: -0.0171911, -0.0156491, -0.0171911, -0.0156491, -0.0015420, 0.0015420
4: 0.0197616, 0.0211237, 0.0197616, 0.0211237, -0.0013621, 0.0013621

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33

Time for candidate selection: 0.20 seconds

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

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33

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
time: 0.32 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: 0.0195159, 0.0227596, 0.0159617, 0.0316399, -0.0121241, 0.0067978
1: -0.0215229, -0.0212697, -0.0219832, -0.0210485, -0.0004744, 0.0007135
2: 0.0186858, 0.0191737, 0.0186291, 0.0196666, -0.0009807, 0.0005445
3: -0.0170585, -0.0163813, -0.0171911, -0.0156491, -0.0014094, 0.0008097
4: 0.0198993, 0.0204838, 0.0197616, 0.0211237, -0.0012244, 0.0007222

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206569, upper bound: 0.0204068
time: 0.33 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206458, upper bound: 0.0205128
time: 0.32 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: 0.0195159, 0.0227596, 0.0195159, 0.0227596, -0.0032437, 0.0032437
1: -0.0215229, -0.0212697, -0.0215229, -0.0212697, -0.0002532, 0.0002532
2: 0.0186858, 0.0191737, 0.0186858, 0.0191737, -0.0004878, 0.0004878
3: -0.0170585, -0.0163813, -0.0170585, -0.0163813, -0.0006772, 0.0006772
4: 0.0198993, 0.0204838, 0.0198993, 0.0204838, -0.0005845, 0.0005845

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206569, upper bound: 0.0205125
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206458, upper bound: 0.0205514
time: 0.32 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.50 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.50
Output dim: 0, lower bound: -0.0205139, upper bound: 0.0203518
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.50
Output dim: 0, lower bound: -0.0203642, upper bound: 0.0203642
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.50
Output dim: 0, lower bound: -0.0205139, upper bound: 0.0206435
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.50
Output dim: 0, lower bound: -0.0203642, upper bound: 0.0206439
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.50
Output dim: 0, lower bound: -0.0206569, upper bound: 0.0204068
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.50
Output dim: 0, lower bound: -0.0206458, upper bound: 0.0205128
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.50
Output dim: 0, lower bound: -0.0206569, upper bound: 0.0205125
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.50
Output dim: 0, lower bound: -0.0206458, upper bound: 0.0205514

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0162792, 0.0311817, 0.0159617, 0.0316399, -0.0153608, 0.0152200
1: -0.0219515, -0.0210579, -0.0219832, -0.0210485, -0.0009029, 0.0009253
2: 0.0186381, 0.0196478, 0.0186291, 0.0196666, -0.0010284, 0.0010186
3: -0.0171801, -0.0156917, -0.0171911, -0.0156491, -0.0015310, 0.0014993
4: 0.0197730, 0.0210815, 0.0197616, 0.0211237, -0.0013508, 0.0013199

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

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
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0151246, 0.0300819, 0.0159617, 0.0316399, -0.0165153, 0.0141201
1: -0.0218969, -0.0210807, -0.0219832, -0.0210485, -0.0008484, 0.0009025
2: 0.0186162, 0.0196312, 0.0186291, 0.0196666, -0.0010503, 0.0010021
3: -0.0172251, -0.0157061, -0.0171911, -0.0156491, -0.0015759, 0.0014849
4: 0.0197272, 0.0210690, 0.0197616, 0.0211237, -0.0013965, 0.0013074

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

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

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

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
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0151246, 0.0300819, 0.0195159, 0.0227596, -0.0076349, 0.0105660
1: -0.0218969, -0.0210807, -0.0215229, -0.0212697, -0.0006272, 0.0004422
2: 0.0186162, 0.0196312, 0.0186858, 0.0191737, -0.0005574, 0.0009454
3: -0.0172251, -0.0157061, -0.0170585, -0.0163813, -0.0008437, 0.0013524
4: 0.0197272, 0.0210690, 0.0198993, 0.0204838, -0.0007566, 0.0011697

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0202414, upper bound: 0.0206294
time: 0.32 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0203474, upper bound: 0.0206064
time: 0.33 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0208043, 0.0225723, 0.0166134, 0.0315257, -0.0107215, 0.0059590
1: -0.0214983, -0.0212676, -0.0219673, -0.0210644, -0.0004339, 0.0006997
2: 0.0187086, 0.0191650, 0.0186437, 0.0196299, -0.0009212, 0.0005213
3: -0.0170065, -0.0163937, -0.0171661, -0.0157149, -0.0012916, 0.0007724
4: 0.0199520, 0.0204754, 0.0197873, 0.0210626, -0.0011106, 0.0006881

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206294, upper bound: 0.0203911
time: 0.32 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206294, upper bound: 0.0202414
time: 0.32 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0196407, 0.0227297, 0.0159617, 0.0316399, -0.0119992, 0.0067680
1: -0.0215221, -0.0212729, -0.0219832, -0.0210485, -0.0004736, 0.0007103
2: 0.0186879, 0.0191708, 0.0186291, 0.0196666, -0.0009786, 0.0005417
3: -0.0170537, -0.0163849, -0.0171911, -0.0156491, -0.0014045, 0.0008062
4: 0.0199043, 0.0204809, 0.0197616, 0.0211237, -0.0012194, 0.0007193

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206003, upper bound: 0.0204971
time: 0.33 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206064, upper bound: 0.0203474
time: 0.32 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0208043, 0.0225723, 0.0201103, 0.0227122, -0.0019079, 0.0024621
1: -0.0214983, -0.0212676, -0.0215193, -0.0212801, -0.0002182, 0.0002517
2: 0.0187086, 0.0191650, 0.0186972, 0.0191613, -0.0004527, 0.0004678
3: -0.0170065, -0.0163937, -0.0170334, -0.0163970, -0.0006095, 0.0006397
4: 0.0199520, 0.0204754, 0.0199245, 0.0204709, -0.0005190, 0.0005508

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0205547, upper bound: 0.0204037
time: 0.33 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0205547, upper bound: 0.0205125
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0196407, 0.0227297, 0.0195159, 0.0227596, -0.0031189, 0.0032138
1: -0.0215221, -0.0212729, -0.0215229, -0.0212697, -0.0002524, 0.0002500
2: 0.0186879, 0.0191708, 0.0186858, 0.0191737, -0.0004857, 0.0004850
3: -0.0170537, -0.0163849, -0.0170585, -0.0163813, -0.0006723, 0.0006736
4: 0.0199043, 0.0204809, 0.0198993, 0.0204838, -0.0005796, 0.0005816

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0205547, upper bound: 0.0204037
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0205547, upper bound: 0.0205514
time: 0.32 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.53 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.53
Output dim: 0, lower bound: -0.0203518, upper bound: 0.0203518
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.53
Output dim: 0, lower bound: -0.0203518, upper bound: 0.0203518
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.53
Output dim: 0, lower bound: -0.0203518, upper bound: 0.0203642
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.53
Output dim: 0, lower bound: -0.0203518, upper bound: 0.0203642
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.53
Output dim: 0, lower bound: -0.0203911, upper bound: 0.0206294
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.53
Output dim: 0, lower bound: -0.0204971, upper bound: 0.0206003
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.53
Output dim: 0, lower bound: -0.0202414, upper bound: 0.0206294
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.53
Output dim: 0, lower bound: -0.0203474, upper bound: 0.0206064
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.53
Output dim: 0, lower bound: -0.0206294, upper bound: 0.0203911
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.53
Output dim: 0, lower bound: -0.0206294, upper bound: 0.0202414
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.53
Output dim: 0, lower bound: -0.0206003, upper bound: 0.0204971
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.53
Output dim: 0, lower bound: -0.0206064, upper bound: 0.0203474
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.53
Output dim: 0, lower bound: -0.0205547, upper bound: 0.0204037
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.53
Output dim: 0, lower bound: -0.0205547, upper bound: 0.0205125
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.53
Output dim: 0, lower bound: -0.0205547, upper bound: 0.0204037
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.53
Output dim: 0, lower bound: -0.0205547, upper bound: 0.0205514

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0162792, 0.0311817, 0.0162792, 0.0311817, -0.0149025, 0.0149025
1: -0.0219515, -0.0210579, -0.0219515, -0.0210579, -0.0008935, 0.0008935
2: 0.0186381, 0.0196478, 0.0186381, 0.0196478, -0.0010096, 0.0010096
3: -0.0171801, -0.0156917, -0.0171801, -0.0156917, -0.0014884, 0.0014884
4: 0.0197730, 0.0210815, 0.0197730, 0.0210815, -0.0013085, 0.0013085

Time for backsubstitution: 2.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0162792, 0.0311817, 0.0151246, 0.0300819, -0.0138027, 0.0160571
1: -0.0219515, -0.0210579, -0.0218969, -0.0210807, -0.0008708, 0.0008390
2: 0.0186381, 0.0196478, 0.0186162, 0.0196312, -0.0009931, 0.0010315
3: -0.0171801, -0.0156917, -0.0172251, -0.0157061, -0.0014740, 0.0015333
4: 0.0197730, 0.0210815, 0.0197272, 0.0210690, -0.0012960, 0.0013543

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0151246, 0.0300819, 0.0162792, 0.0311817, -0.0160571, 0.0138027
1: -0.0218969, -0.0210807, -0.0219515, -0.0210579, -0.0008390, 0.0008708
2: 0.0186162, 0.0196312, 0.0186381, 0.0196478, -0.0010315, 0.0009931
3: -0.0172251, -0.0157061, -0.0171801, -0.0156917, -0.0015333, 0.0014740
4: 0.0197272, 0.0210690, 0.0197730, 0.0210815, -0.0013543, 0.0012960

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0151246, 0.0300819, 0.0151246, 0.0300819, -0.0149572, 0.0149572
1: -0.0218969, -0.0210807, -0.0218969, -0.0210807, -0.0008162, 0.0008162
2: 0.0186162, 0.0196312, 0.0186162, 0.0196312, -0.0010150, 0.0010150
3: -0.0172251, -0.0157061, -0.0172251, -0.0157061, -0.0015189, 0.0015189
4: 0.0197272, 0.0210690, 0.0197272, 0.0210690, -0.0013418, 0.0013418

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0168856, 0.0310751, 0.0208043, 0.0225723, -0.0056868, 0.0102708
1: -0.0219408, -0.0210716, -0.0214983, -0.0212676, -0.0006732, 0.0004267
2: 0.0186480, 0.0196156, 0.0187086, 0.0191650, -0.0005170, 0.0009070
3: -0.0171560, -0.0157544, -0.0170065, -0.0163937, -0.0007623, 0.0012521
4: 0.0197978, 0.0210292, 0.0199520, 0.0204754, -0.0006776, 0.0010772

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0203010, upper bound: 0.0204001
time: 0.32 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2
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

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0162792, 0.0311817, 0.0196407, 0.0227297, -0.0064505, 0.0115410
1: -0.0219515, -0.0210579, -0.0215221, -0.0212729, -0.0006786, 0.0004642
2: 0.0186381, 0.0196478, 0.0186879, 0.0191708, -0.0005327, 0.0009598
3: -0.0171801, -0.0156917, -0.0170537, -0.0163849, -0.0007952, 0.0013619
4: 0.0197730, 0.0210815, 0.0199043, 0.0204809, -0.0007079, 0.0011772

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0193171, upper bound: 0.0205934
time: 0.32 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0157696, 0.0299950, 0.0208043, 0.0225723, -0.0068027, 0.0091907
1: -0.0218877, -0.0210942, -0.0214983, -0.0212676, -0.0006201, 0.0004041
2: 0.0186278, 0.0195903, 0.0187086, 0.0191650, -0.0005372, 0.0008817
3: -0.0171995, -0.0157616, -0.0170065, -0.0163937, -0.0008058, 0.0012449
4: 0.0197530, 0.0210225, 0.0199520, 0.0204754, -0.0007224, 0.0010706

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0201514, upper bound: 0.0203998
time: 0.32 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0192672, upper bound: 0.0206284
time: 0.33 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0151246, 0.0300819, 0.0196407, 0.0227297, -0.0076051, 0.0104412
1: -0.0218969, -0.0210807, -0.0215221, -0.0212729, -0.0006240, 0.0004414
2: 0.0186162, 0.0196312, 0.0186879, 0.0191708, -0.0005546, 0.0009433
3: -0.0172251, -0.0157061, -0.0170537, -0.0163849, -0.0008402, 0.0013475
4: 0.0197272, 0.0210690, 0.0199043, 0.0204809, -0.0007537, 0.0011647

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0191674, upper bound: 0.0206030
time: 0.32 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0208043, 0.0225723, 0.0168856, 0.0310751, -0.0102708, 0.0056868
1: -0.0214983, -0.0212676, -0.0219408, -0.0210716, -0.0004267, 0.0006732
2: 0.0187086, 0.0191650, 0.0186480, 0.0196156, -0.0009070, 0.0005170
3: -0.0170065, -0.0163937, -0.0171560, -0.0157544, -0.0012521, 0.0007623
4: 0.0199520, 0.0204754, 0.0197978, 0.0210292, -0.0010772, 0.0006776

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206112, upper bound: 0.0195850
time: 0.33 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206112, upper bound: 0.0202414
time: 0.33 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0208043, 0.0225723, 0.0157696, 0.0299950, -0.0091907, 0.0068027
1: -0.0214983, -0.0212676, -0.0218877, -0.0210942, -0.0004041, 0.0006201
2: 0.0187086, 0.0191650, 0.0186278, 0.0195903, -0.0008817, 0.0005372
3: -0.0170065, -0.0163937, -0.0171995, -0.0157616, -0.0012449, 0.0008058
4: 0.0199520, 0.0204754, 0.0197530, 0.0210225, -0.0010706, 0.0007224

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206146, upper bound: 0.0195850
time: 0.34 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206146, upper bound: 0.0202414
time: 0.33 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0196407, 0.0227297, 0.0162792, 0.0311817, -0.0115410, 0.0064505
1: -0.0215221, -0.0212729, -0.0219515, -0.0210579, -0.0004642, 0.0006786
2: 0.0186879, 0.0191708, 0.0186381, 0.0196478, -0.0009598, 0.0005327
3: -0.0170537, -0.0163849, -0.0171801, -0.0156917, -0.0013619, 0.0007952
4: 0.0199043, 0.0204809, 0.0197730, 0.0210815, -0.0011772, 0.0007079

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206003, upper bound: 0.0203377
time: 0.32 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206003, upper bound: 0.0203474
time: 0.33 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0196407, 0.0227297, 0.0151246, 0.0300819, -0.0104412, 0.0076051
1: -0.0215221, -0.0212729, -0.0218969, -0.0210807, -0.0004414, 0.0006240
2: 0.0186879, 0.0191708, 0.0186162, 0.0196312, -0.0009433, 0.0005546
3: -0.0170537, -0.0163849, -0.0172251, -0.0157061, -0.0013475, 0.0008402
4: 0.0199043, 0.0204809, 0.0197272, 0.0210690, -0.0011647, 0.0007537

Time for backsubstitution: 2.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206064, upper bound: 0.0203377
time: 0.33 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206064, upper bound: 0.0203474
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0208043, 0.0225723, 0.0208043, 0.0225723, -0.0017681, 0.0017681
1: -0.0214983, -0.0212676, -0.0214983, -0.0212676, -0.0002307, 0.0002307
2: 0.0187086, 0.0191650, 0.0187086, 0.0191650, -0.0004564, 0.0004564
3: -0.0170065, -0.0163937, -0.0170065, -0.0163937, -0.0006128, 0.0006128
4: 0.0199520, 0.0204754, 0.0199520, 0.0204754, -0.0005234, 0.0005234

Time for backsubstitution: 2.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0205286, upper bound: 0.0199270
time: 0.33 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0205554, upper bound: 0.0204429
time: 0.33 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0208043, 0.0225723, 0.0196407, 0.0227297, -0.0019254, 0.0029316
1: -0.0214983, -0.0212676, -0.0215221, -0.0212729, -0.0002254, 0.0002545
2: 0.0187086, 0.0191650, 0.0186879, 0.0191708, -0.0004622, 0.0004771
3: -0.0170065, -0.0163937, -0.0170537, -0.0163849, -0.0006216, 0.0006600
4: 0.0199520, 0.0204754, 0.0199043, 0.0204809, -0.0005289, 0.0005711

Time for backsubstitution: 2.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0205286, upper bound: 0.0199270
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0205554, upper bound: 0.0204657
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0196407, 0.0227297, 0.0208043, 0.0225723, -0.0029316, 0.0019254
1: -0.0215221, -0.0212729, -0.0214983, -0.0212676, -0.0002545, 0.0002254
2: 0.0186879, 0.0191708, 0.0187086, 0.0191650, -0.0004771, 0.0004622
3: -0.0170537, -0.0163849, -0.0170065, -0.0163937, -0.0006600, 0.0006216
4: 0.0199043, 0.0204809, 0.0199520, 0.0204754, -0.0005711, 0.0005289

Time for backsubstitution: 2.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0205213, upper bound: 0.0201907
time: 0.35 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0205040, upper bound: 0.0203660
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0196407, 0.0227297, 0.0196407, 0.0227297, -0.0030890, 0.0030890
1: -0.0215221, -0.0212729, -0.0215221, -0.0212729, -0.0002492, 0.0002492
2: 0.0186879, 0.0191708, 0.0186879, 0.0191708, -0.0004829, 0.0004829
3: -0.0170537, -0.0163849, -0.0170537, -0.0163849, -0.0006688, 0.0006688
4: 0.0199043, 0.0204809, 0.0199043, 0.0204809, -0.0005766, 0.0005766

Time for backsubstitution: 2.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

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
- Time for NS candidates: 4.12 seconds
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.12
Output dim: 0, lower bound: -0.0206112, upper bound: 0.0195850
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.12
Output dim: 0, lower bound: -0.0206112, upper bound: 0.0202414
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.12
Output dim: 0, lower bound: -0.0206146, upper bound: 0.0195850
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.12
Output dim: 0, lower bound: -0.0206146, upper bound: 0.0202414
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.12
Output dim: 0, lower bound: -0.0206003, upper bound: 0.0203377
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.12
Output dim: 0, lower bound: -0.0206003, upper bound: 0.0203474
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.12
Output dim: 0, lower bound: -0.0206064, upper bound: 0.0203377
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.12
Output dim: 0, lower bound: -0.0206064, upper bound: 0.0203474
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.12
Output dim: 0, lower bound: -0.0205286, upper bound: 0.0199270
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.12
Output dim: 0, lower bound: -0.0205554, upper bound: 0.0204429
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.12
Output dim: 0, lower bound: -0.0205286, upper bound: 0.0199270
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.12
Output dim: 0, lower bound: -0.0205554, upper bound: 0.0204657
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.12
Output dim: 0, lower bound: -0.0205213, upper bound: 0.0201907
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.12
Output dim: 0, lower bound: -0.0205040, upper bound: 0.0203660
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.12
Output dim: 0, lower bound: -0.0205213, upper bound: 0.0203943
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.12
Output dim: 0, lower bound: -0.0205040, upper bound: 0.0203939

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0209451, 0.0225421, 0.0168856, 0.0310751, -0.0101300, 0.0056565
1: -0.0214969, -0.0212741, -0.0219408, -0.0210716, -0.0004253, 0.0006667
2: 0.0187107, 0.0191514, 0.0186480, 0.0196156, -0.0009050, 0.0005033
3: -0.0170011, -0.0164111, -0.0171560, -0.0157544, -0.0012467, 0.0007449
4: 0.0199576, 0.0204610, 0.0197978, 0.0210292, -0.0010716, 0.0006632

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206102, upper bound: 0.0194169
time: 0.32 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0210209, 0.0225073, 0.0168856, 0.0310751, -0.0100542, 0.0056217
1: -0.0214985, -0.0212876, -0.0219408, -0.0210716, -0.0004269, 0.0006532
2: 0.0187120, 0.0191457, 0.0186480, 0.0196156, -0.0009036, 0.0004977
3: -0.0169966, -0.0164179, -0.0171560, -0.0157544, -0.0012422, 0.0007381
4: 0.0199614, 0.0204549, 0.0197978, 0.0210292, -0.0010678, 0.0006571

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206102, upper bound: 0.0194169
time: 0.32 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0209451, 0.0225421, 0.0157696, 0.0299950, -0.0090499, 0.0067725
1: -0.0214969, -0.0212741, -0.0218877, -0.0210942, -0.0004027, 0.0006136
2: 0.0187107, 0.0191514, 0.0186278, 0.0195903, -0.0008796, 0.0005236
3: -0.0170011, -0.0164111, -0.0171995, -0.0157616, -0.0012395, 0.0007885
4: 0.0199576, 0.0204610, 0.0197530, 0.0210225, -0.0010649, 0.0007080

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206102, upper bound: 0.0192672
time: 0.33 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0210209, 0.0225073, 0.0157696, 0.0299950, -0.0089740, 0.0067377
1: -0.0214985, -0.0212876, -0.0218877, -0.0210942, -0.0004043, 0.0006001
2: 0.0187120, 0.0191457, 0.0186278, 0.0195903, -0.0008782, 0.0005180
3: -0.0169966, -0.0164179, -0.0171995, -0.0157616, -0.0012350, 0.0007817
4: 0.0199614, 0.0204549, 0.0197530, 0.0210225, -0.0010611, 0.0007019

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206102, upper bound: 0.0192672
time: 0.33 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0198755, 0.0226901, 0.0162792, 0.0311817, -0.0113062, 0.0064109
1: -0.0215207, -0.0212857, -0.0219515, -0.0210579, -0.0004627, 0.0006658
2: 0.0186918, 0.0191581, 0.0186381, 0.0196478, -0.0009560, 0.0005200
3: -0.0170443, -0.0164011, -0.0171801, -0.0156917, -0.0013525, 0.0007790
4: 0.0199138, 0.0204675, 0.0197730, 0.0210815, -0.0011677, 0.0006945

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0205934, upper bound: 0.0193171
time: 0.33 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0185295, 0.0226950, 0.0162792, 0.0311817, -0.0126522, 0.0064158
1: -0.0215215, -0.0212783, -0.0219515, -0.0210579, -0.0004636, 0.0006732
2: 0.0186682, 0.0191596, 0.0186381, 0.0196478, -0.0009796, 0.0005215
3: -0.0170986, -0.0163983, -0.0171801, -0.0156917, -0.0014068, 0.0007818
4: 0.0198591, 0.0204693, 0.0197730, 0.0210815, -0.0012224, 0.0006963

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0205934, upper bound: 0.0193171
time: 0.33 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0198755, 0.0226901, 0.0151246, 0.0300819, -0.0102064, 0.0075654
1: -0.0215207, -0.0212857, -0.0218969, -0.0210807, -0.0004400, 0.0006113
2: 0.0186918, 0.0191581, 0.0186162, 0.0196312, -0.0009394, 0.0005419
3: -0.0170443, -0.0164011, -0.0172251, -0.0157061, -0.0013381, 0.0008240
4: 0.0199138, 0.0204675, 0.0197272, 0.0210690, -0.0011552, 0.0007403

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0205934, upper bound: 0.0191674
time: 0.32 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0185295, 0.0226950, 0.0151246, 0.0300819, -0.0115524, 0.0075703
1: -0.0215215, -0.0212783, -0.0218969, -0.0210807, -0.0004408, 0.0006186
2: 0.0186682, 0.0191596, 0.0186162, 0.0196312, -0.0009630, 0.0005434
3: -0.0170986, -0.0163983, -0.0172251, -0.0157061, -0.0013924, 0.0008268
4: 0.0198591, 0.0204693, 0.0197272, 0.0210690, -0.0012099, 0.0007421

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0205934, upper bound: 0.0191674
time: 0.33 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0209451, 0.0225421, 0.0208043, 0.0225723, -0.0016273, 0.0017378
1: -0.0214969, -0.0212741, -0.0214983, -0.0212676, -0.0002293, 0.0002242
2: 0.0187107, 0.0191514, 0.0187086, 0.0191650, -0.0004543, 0.0004427
3: -0.0170011, -0.0164111, -0.0170065, -0.0163937, -0.0006074, 0.0005954
4: 0.0199576, 0.0204610, 0.0199520, 0.0204754, -0.0005178, 0.0005090

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199516, upper bound: 0.0199291
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199516, upper bound: 0.0199291
time: 0.32 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0210209, 0.0225073, 0.0208043, 0.0225723, -0.0015514, 0.0017030
1: -0.0214985, -0.0212876, -0.0214983, -0.0212676, -0.0002309, 0.0002107
2: 0.0187120, 0.0191457, 0.0187086, 0.0191650, -0.0004530, 0.0004371
3: -0.0169966, -0.0164179, -0.0170065, -0.0163937, -0.0006029, 0.0005886
4: 0.0199614, 0.0204549, 0.0199520, 0.0204754, -0.0005140, 0.0005030

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199783, upper bound: 0.0204283
time: 0.33 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199783, upper bound: 0.0204429
time: 0.32 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0209451, 0.0225421, 0.0196407, 0.0227297, -0.0017846, 0.0029014
1: -0.0214969, -0.0212741, -0.0215221, -0.0212729, -0.0002240, 0.0002480
2: 0.0187107, 0.0191514, 0.0186879, 0.0191708, -0.0004602, 0.0004634
3: -0.0170011, -0.0164111, -0.0170537, -0.0163849, -0.0006162, 0.0006426
4: 0.0199576, 0.0204610, 0.0199043, 0.0204809, -0.0005233, 0.0005567

Time for backsubstitution: 2.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206112, upper bound: 0.0199270
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206112, upper bound: 0.0199270
time: 0.35 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0210209, 0.0225073, 0.0196407, 0.0227297, -0.0017088, 0.0028666
1: -0.0214985, -0.0212876, -0.0215221, -0.0212729, -0.0002256, 0.0002345
2: 0.0187120, 0.0191457, 0.0186879, 0.0191708, -0.0004588, 0.0004578
3: -0.0169966, -0.0164179, -0.0170537, -0.0163849, -0.0006117, 0.0006358
4: 0.0199614, 0.0204549, 0.0199043, 0.0204809, -0.0005195, 0.0005506

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0204175, upper bound: 0.0203989
time: 0.33 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206323, upper bound: 0.0204657
time: 0.33 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0206323, upper bound: 0.0204657
time: 0.33 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0198755, 0.0226901, 0.0208043, 0.0225723, -0.0026969, 0.0018858
1: -0.0215207, -0.0212857, -0.0214983, -0.0212676, -0.0002531, 0.0002126
2: 0.0186918, 0.0191581, 0.0187086, 0.0191650, -0.0004732, 0.0004495
3: -0.0170443, -0.0164011, -0.0170065, -0.0163937, -0.0006506, 0.0006054
4: 0.0199138, 0.0204675, 0.0199520, 0.0204754, -0.0005616, 0.0005155

Time for backsubstitution: 2.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199118, upper bound: 0.0201021
time: 0.33 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199118, upper bound: 0.0201907
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0185295, 0.0226950, 0.0208043, 0.0225723, -0.0040429, 0.0018907
1: -0.0215215, -0.0212783, -0.0214983, -0.0212676, -0.0002539, 0.0002200
2: 0.0186682, 0.0191596, 0.0187086, 0.0191650, -0.0004968, 0.0004510
3: -0.0170986, -0.0163983, -0.0170065, -0.0163937, -0.0007049, 0.0006082
4: 0.0198591, 0.0204693, 0.0199520, 0.0204754, -0.0006163, 0.0005173

Time for backsubstitution: 2.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199118, upper bound: 0.0202359
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0199118, upper bound: 0.0203660
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0198755, 0.0226901, 0.0196407, 0.0227297, -0.0028542, 0.0030493
1: -0.0215207, -0.0212857, -0.0215221, -0.0212729, -0.0002478, 0.0002364
2: 0.0186918, 0.0191581, 0.0186879, 0.0191708, -0.0004790, 0.0004702
3: -0.0170443, -0.0164011, -0.0170537, -0.0163849, -0.0006594, 0.0006526
4: 0.0199138, 0.0204675, 0.0199043, 0.0204809, -0.0005671, 0.0005632

Time for backsubstitution: 2.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

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

Time for backsubstitution: 2.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0203989, upper bound: 0.0203906
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0203989, upper bound: 0.0203939
time: 0.34 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.59 seconds
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 0, lower bound: -0.0199516, upper bound: 0.0199291
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 0, lower bound: -0.0199516, upper bound: 0.0199291
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 0, lower bound: -0.0199783, upper bound: 0.0204283
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 0, lower bound: -0.0199783, upper bound: 0.0204429
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 0, lower bound: -0.0206112, upper bound: 0.0199270
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 0, lower bound: -0.0206112, upper bound: 0.0199270
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 0, lower bound: -0.0206323, upper bound: 0.0204657
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 0, lower bound: -0.0206323, upper bound: 0.0204657
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 0, lower bound: -0.0199118, upper bound: 0.0201021
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 0, lower bound: -0.0199118, upper bound: 0.0201907
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 0, lower bound: -0.0199118, upper bound: 0.0202359
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 0, lower bound: -0.0199118, upper bound: 0.0203660
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 0, lower bound: -0.0203989, upper bound: 0.0203906
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 0, lower bound: -0.0203989, upper bound: 0.0203939
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 0, lower bound: -0.0203989, upper bound: 0.0203906
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 0, lower bound: -0.0203989, upper bound: 0.0203939

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0209451, 0.0225421, 0.0209451, 0.0225421, -0.0015970, 0.0015970
1: -0.0214969, -0.0212741, -0.0214969, -0.0212741, -0.0002227, 0.0002227
2: 0.0187107, 0.0191514, 0.0187107, 0.0191514, -0.0004407, 0.0004407
3: -0.0170011, -0.0164111, -0.0170011, -0.0164111, -0.0005901, 0.0005901
4: 0.0199576, 0.0204610, 0.0199576, 0.0204610, -0.0005034, 0.0005034

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0195544, upper bound: 0.0199176
time: 0.32 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0195544, upper bound: 0.0195725
time: 0.33 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0209451, 0.0225421, 0.0210209, 0.0225073, -0.0015622, 0.0015212
1: -0.0214969, -0.0212741, -0.0214985, -0.0212876, -0.0002093, 0.0002244
2: 0.0187107, 0.0191514, 0.0187120, 0.0191457, -0.0004351, 0.0004393
3: -0.0170011, -0.0164111, -0.0169966, -0.0164179, -0.0005833, 0.0005856
4: 0.0199576, 0.0204610, 0.0199614, 0.0204549, -0.0004973, 0.0004996

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0195544, upper bound: 0.0199176
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0195544, upper bound: 0.0195743
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0210209, 0.0225073, 0.0209451, 0.0225421, -0.0015212, 0.0015622
1: -0.0214985, -0.0212876, -0.0214969, -0.0212741, -0.0002244, 0.0002093
2: 0.0187120, 0.0191457, 0.0187107, 0.0191514, -0.0004393, 0.0004351
3: -0.0169966, -0.0164179, -0.0170011, -0.0164111, -0.0005856, 0.0005833
4: 0.0199614, 0.0204549, 0.0199576, 0.0204610, -0.0004996, 0.0004973

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0195811, upper bound: 0.0204269
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0195811, upper bound: 0.0195299
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0210209, 0.0225073, 0.0210209, 0.0225073, -0.0014864, 0.0014864
1: -0.0214985, -0.0212876, -0.0214985, -0.0212876, -0.0002109, 0.0002109
2: 0.0187120, 0.0191457, 0.0187120, 0.0191457, -0.0004337, 0.0004337
3: -0.0169966, -0.0164179, -0.0169966, -0.0164179, -0.0005787, 0.0005787
4: 0.0199614, 0.0204549, 0.0199614, 0.0204549, -0.0004935, 0.0004935

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0195811, upper bound: 0.0204332
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0195811, upper bound: 0.0195299
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0209451, 0.0225421, 0.0198755, 0.0226901, -0.0017450, 0.0026666
1: -0.0214969, -0.0212741, -0.0215207, -0.0212857, -0.0002112, 0.0002465
2: 0.0187107, 0.0191514, 0.0186918, 0.0191581, -0.0004475, 0.0004596
3: -0.0170011, -0.0164111, -0.0170443, -0.0164011, -0.0006001, 0.0006332
4: 0.0199576, 0.0204610, 0.0199138, 0.0204675, -0.0005099, 0.0005472

Time for backsubstitution: 2.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0194546, upper bound: 0.0199128
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0194546, upper bound: 0.0195741
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0209451, 0.0225421, 0.0185295, 0.0226950, -0.0017499, 0.0040126
1: -0.0214969, -0.0212741, -0.0215215, -0.0212783, -0.0002186, 0.0002474
2: 0.0187107, 0.0191514, 0.0186682, 0.0191596, -0.0004490, 0.0004832
3: -0.0170011, -0.0164111, -0.0170986, -0.0163983, -0.0006028, 0.0006875
4: 0.0199576, 0.0204610, 0.0198591, 0.0204693, -0.0005117, 0.0006019

Time for backsubstitution: 2.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0194546, upper bound: 0.0199128
time: 0.33 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0194546, upper bound: 0.0195743
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0210209, 0.0225073, 0.0198755, 0.0226901, -0.0016691, 0.0026318
1: -0.0214985, -0.0212876, -0.0215207, -0.0212857, -0.0002128, 0.0002331
2: 0.0187120, 0.0191457, 0.0186918, 0.0191581, -0.0004461, 0.0004539
3: -0.0169966, -0.0164179, -0.0170443, -0.0164011, -0.0005955, 0.0006264
4: 0.0199614, 0.0204549, 0.0199138, 0.0204675, -0.0005061, 0.0005411

Time for backsubstitution: 2.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0194813, upper bound: 0.0204402
time: 0.35 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0194813, upper bound: 0.0195239
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0210209, 0.0225073, 0.0185295, 0.0226950, -0.0016740, 0.0039778
1: -0.0214985, -0.0212876, -0.0215215, -0.0212783, -0.0002202, 0.0002339
2: 0.0187120, 0.0191457, 0.0186682, 0.0191596, -0.0004476, 0.0004775
3: -0.0169966, -0.0164179, -0.0170986, -0.0163983, -0.0005983, 0.0006807
4: 0.0199614, 0.0204549, 0.0198591, 0.0204693, -0.0005079, 0.0005958

Time for backsubstitution: 2.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0194813, upper bound: 0.0204402
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0194813, upper bound: 0.0195239
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0198755, 0.0226901, 0.0209451, 0.0225421, -0.0026666, 0.0017450
1: -0.0215207, -0.0212857, -0.0214969, -0.0212741, -0.0002465, 0.0002112
2: 0.0186918, 0.0191581, 0.0187107, 0.0191514, -0.0004596, 0.0004475
3: -0.0170443, -0.0164011, -0.0170011, -0.0164111, -0.0006332, 0.0006001
4: 0.0199138, 0.0204675, 0.0199576, 0.0204610, -0.0005472, 0.0005099

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0195471, upper bound: 0.0200917
time: 0.35 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0195471, upper bound: 0.0195762
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0198755, 0.0226901, 0.0210209, 0.0225073, -0.0026318, 0.0016691
1: -0.0215207, -0.0212857, -0.0214985, -0.0212876, -0.0002331, 0.0002128
2: 0.0186918, 0.0191581, 0.0187120, 0.0191457, -0.0004539, 0.0004461
3: -0.0170443, -0.0164011, -0.0169966, -0.0164179, -0.0006264, 0.0005955
4: 0.0199138, 0.0204675, 0.0199614, 0.0204549, -0.0005411, 0.0005061

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0195471, upper bound: 0.0201478
time: 0.35 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0195471, upper bound: 0.0195911
time: 0.33 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0185295, 0.0226950, 0.0209451, 0.0225421, -0.0040126, 0.0017499
1: -0.0215215, -0.0212783, -0.0214969, -0.0212741, -0.0002474, 0.0002186
2: 0.0186682, 0.0191596, 0.0187107, 0.0191514, -0.0004832, 0.0004490
3: -0.0170986, -0.0163983, -0.0170011, -0.0164111, -0.0006875, 0.0006028
4: 0.0198591, 0.0204693, 0.0199576, 0.0204610, -0.0006019, 0.0005117

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0195052, upper bound: 0.0202306
time: 0.35 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0194648, upper bound: 0.0193907
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0185295, 0.0226950, 0.0210209, 0.0225073, -0.0039778, 0.0016740
1: -0.0215215, -0.0212783, -0.0214985, -0.0212876, -0.0002339, 0.0002202
2: 0.0186682, 0.0191596, 0.0187120, 0.0191457, -0.0004775, 0.0004476
3: -0.0170986, -0.0163983, -0.0169966, -0.0164179, -0.0006807, 0.0005983
4: 0.0198591, 0.0204693, 0.0199614, 0.0204549, -0.0005958, 0.0005079

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0195052, upper bound: 0.0202675
time: 0.35 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0194648, upper bound: 0.0193907
time: 0.35 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0198755, 0.0226901, 0.0198755, 0.0226901, -0.0028146, 0.0028146
1: -0.0215207, -0.0212857, -0.0215207, -0.0212857, -0.0002350, 0.0002350
2: 0.0186918, 0.0191581, 0.0186918, 0.0191581, -0.0004663, 0.0004663
3: -0.0170443, -0.0164011, -0.0170443, -0.0164011, -0.0006432, 0.0006432
4: 0.0199138, 0.0204675, 0.0199138, 0.0204675, -0.0005537, 0.0005537

Time for backsubstitution: 2.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

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

Time for backsubstitution: 2.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0194473, upper bound: 0.0201925
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0194473, upper bound: 0.0196157
time: 0.33 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0185295, 0.0226950, 0.0198755, 0.0226901, -0.0041606, 0.0028195
1: -0.0215215, -0.0212783, -0.0215207, -0.0212857, -0.0002359, 0.0002424
2: 0.0186682, 0.0191596, 0.0186918, 0.0191581, -0.0004899, 0.0004678
3: -0.0170986, -0.0163983, -0.0170443, -0.0164011, -0.0006975, 0.0006460
4: 0.0198591, 0.0204693, 0.0199138, 0.0204675, -0.0006084, 0.0005555

Time for backsubstitution: 2.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0194300, upper bound: 0.0202621
time: 0.35 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0194241, upper bound: 0.0193906
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0185295, 0.0226950, 0.0185295, 0.0226950, -0.0041655, 0.0041655
1: -0.0215215, -0.0212783, -0.0215215, -0.0212783, -0.0002432, 0.0002432
2: 0.0186682, 0.0191596, 0.0186682, 0.0191596, -0.0004914, 0.0004914
3: -0.0170986, -0.0163983, -0.0170986, -0.0163983, -0.0007003, 0.0007003
4: 0.0198591, 0.0204693, 0.0198591, 0.0204693, -0.0006102, 0.0006102

Time for backsubstitution: 2.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0194300, upper bound: 0.0202764
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0194241, upper bound: 0.0193907
time: 0.34 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 3.62 seconds
NS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0195544, upper bound: 0.0199176
NS_A2_B2_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0195544, upper bound: 0.0195725
NS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0195544, upper bound: 0.0199176
NS_A2_B2_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0195544, upper bound: 0.0195743
NS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0195811, upper bound: 0.0204269
NS_A2_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0195811, upper bound: 0.0195299
NS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0195811, upper bound: 0.0204332
NS_A2_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0195811, upper bound: 0.0195299
NS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0194546, upper bound: 0.0199128
NS_A2_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0194546, upper bound: 0.0195741
NS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0194546, upper bound: 0.0199128
NS_A2_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0194546, upper bound: 0.0195743
NS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0194813, upper bound: 0.0204402
NS_A2_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0194813, upper bound: 0.0195239
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0194813, upper bound: 0.0204402
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0194813, upper bound: 0.0195239
NS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0195471, upper bound: 0.0200917
NS_A2_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0195471, upper bound: 0.0195762
NS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0195471, upper bound: 0.0201478
NS_A2_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0195471, upper bound: 0.0195911
NS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0195052, upper bound: 0.0202306
NS_A2_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0194648, upper bound: 0.0193907
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0195052, upper bound: 0.0202675
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0194648, upper bound: 0.0193907
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0194473, upper bound: 0.0201924
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0194473, upper bound: 0.0196157
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0194473, upper bound: 0.0201925
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0194473, upper bound: 0.0196157
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0194300, upper bound: 0.0202621
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0194241, upper bound: 0.0193906
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0194300, upper bound: 0.0202764
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0194241, upper bound: 0.0193907

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0209469, 0.0225406, 0.0209451, 0.0225421, -0.0015952, 0.0015955
1: -0.0214969, -0.0212742, -0.0214969, -0.0212741, -0.0002227, 0.0002227
2: 0.0187107, 0.0191504, 0.0187107, 0.0191514, -0.0004407, 0.0004397
3: -0.0170011, -0.0164123, -0.0170011, -0.0164111, -0.0005900, 0.0005888
4: 0.0199577, 0.0204600, 0.0199576, 0.0204610, -0.0005033, 0.0005024

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0196074, upper bound: 0.0195725
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0196074, upper bound: 0.0195725
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0209469, 0.0225406, 0.0210209, 0.0225073, -0.0015604, 0.0015197
1: -0.0214969, -0.0212742, -0.0214985, -0.0212876, -0.0002092, 0.0002243
2: 0.0187107, 0.0191504, 0.0187120, 0.0191457, -0.0004350, 0.0004384
3: -0.0170011, -0.0164123, -0.0169966, -0.0164179, -0.0005832, 0.0005843
4: 0.0199577, 0.0204600, 0.0199614, 0.0204549, -0.0004973, 0.0004986

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0195544, upper bound: 0.0195743
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0195544, upper bound: 0.0195743
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0210242, 0.0225057, 0.0209451, 0.0225421, -0.0015179, 0.0015606
1: -0.0214985, -0.0212878, -0.0214969, -0.0212741, -0.0002243, 0.0002091
2: 0.0187121, 0.0191445, 0.0187107, 0.0191514, -0.0004393, 0.0004338
3: -0.0169965, -0.0164195, -0.0170011, -0.0164111, -0.0005855, 0.0005817
4: 0.0199615, 0.0204536, 0.0199576, 0.0204610, -0.0004995, 0.0004960

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0196341, upper bound: 0.0195343
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0196341, upper bound: 0.0195343
time: 0.35 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0210242, 0.0225057, 0.0210209, 0.0225073, -0.0014831, 0.0014848
1: -0.0214985, -0.0212878, -0.0214985, -0.0212876, -0.0002109, 0.0002107
2: 0.0187121, 0.0191445, 0.0187120, 0.0191457, -0.0004337, 0.0004324
3: -0.0169965, -0.0164195, -0.0169966, -0.0164179, -0.0005786, 0.0005771
4: 0.0199615, 0.0204536, 0.0199614, 0.0204549, -0.0004934, 0.0004922

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0195811, upper bound: 0.0195299
time: 0.35 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0195811, upper bound: 0.0195299
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0209469, 0.0225406, 0.0198755, 0.0226901, -0.0017432, 0.0026651
1: -0.0214969, -0.0212742, -0.0215207, -0.0212857, -0.0002112, 0.0002464
2: 0.0187107, 0.0191504, 0.0186918, 0.0191581, -0.0004474, 0.0004586
3: -0.0170011, -0.0164123, -0.0170443, -0.0164011, -0.0006000, 0.0006320
4: 0.0199577, 0.0204600, 0.0199138, 0.0204675, -0.0005098, 0.0005462

Time for backsubstitution: 2.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0196749, upper bound: 0.0195853
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0196749, upper bound: 0.0195853
time: 0.33 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0209469, 0.0225406, 0.0185295, 0.0226950, -0.0017481, 0.0040111
1: -0.0214969, -0.0212742, -0.0215215, -0.0212783, -0.0002186, 0.0002473
2: 0.0187107, 0.0191504, 0.0186682, 0.0191596, -0.0004489, 0.0004822
3: -0.0170011, -0.0164123, -0.0170986, -0.0163983, -0.0006028, 0.0006862
4: 0.0199577, 0.0204600, 0.0198591, 0.0204693, -0.0005116, 0.0006009

Time for backsubstitution: 2.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0194546, upper bound: 0.0195743
time: 0.35 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0194546, upper bound: 0.0195743
time: 0.35 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0210242, 0.0225057, 0.0198755, 0.0226901, -0.0016659, 0.0026302
1: -0.0214985, -0.0212878, -0.0215207, -0.0212857, -0.0002128, 0.0002329
2: 0.0187121, 0.0191445, 0.0186918, 0.0191581, -0.0004461, 0.0004527
3: -0.0169965, -0.0164195, -0.0170443, -0.0164011, -0.0005954, 0.0006248
4: 0.0199615, 0.0204536, 0.0199138, 0.0204675, -0.0005060, 0.0005398

Time for backsubstitution: 2.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0197017, upper bound: 0.0195396
time: 0.35 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0197017, upper bound: 0.0195396
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0210242, 0.0225057, 0.0185295, 0.0226950, -0.0016708, 0.0039762
1: -0.0214985, -0.0212878, -0.0215215, -0.0212783, -0.0002202, 0.0002338
2: 0.0187121, 0.0191445, 0.0186682, 0.0191596, -0.0004476, 0.0004763
3: -0.0169965, -0.0164195, -0.0170986, -0.0163983, -0.0005982, 0.0006791
4: 0.0199615, 0.0204536, 0.0198591, 0.0204693, -0.0005078, 0.0005945

Time for backsubstitution: 2.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0194813, upper bound: 0.0195239
time: 0.37 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0194813, upper bound: 0.0195239
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0199707, 0.0226861, 0.0209451, 0.0225421, -0.0025714, 0.0017410
1: -0.0215206, -0.0212867, -0.0214969, -0.0212741, -0.0002465, 0.0002102
2: 0.0186935, 0.0191568, 0.0187107, 0.0191514, -0.0004579, 0.0004461
3: -0.0170405, -0.0164028, -0.0170011, -0.0164111, -0.0006294, 0.0005983
4: 0.0199177, 0.0204661, 0.0199576, 0.0204610, -0.0005433, 0.0005085

Time for backsubstitution: 2.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0196001, upper bound: 0.0195762
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0196001, upper bound: 0.0195762
time: 0.36 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0199707, 0.0226861, 0.0210209, 0.0225073, -0.0025366, 0.0016652
1: -0.0215206, -0.0212867, -0.0214985, -0.0212876, -0.0002330, 0.0002118
2: 0.0186935, 0.0191568, 0.0187120, 0.0191457, -0.0004523, 0.0004447
3: -0.0170405, -0.0164028, -0.0169966, -0.0164179, -0.0006226, 0.0005938
4: 0.0199177, 0.0204661, 0.0199614, 0.0204549, -0.0005373, 0.0005046

Time for backsubstitution: 2.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0195471, upper bound: 0.0195911
time: 0.35 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0195471, upper bound: 0.0195911
time: 0.35 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0186350, 0.0226922, 0.0209451, 0.0225421, -0.0039071, 0.0017471
1: -0.0215215, -0.0212791, -0.0214969, -0.0212741, -0.0002473, 0.0002178
2: 0.0186700, 0.0191581, 0.0187107, 0.0191514, -0.0004814, 0.0004475
3: -0.0170943, -0.0164003, -0.0170011, -0.0164111, -0.0006832, 0.0006009
4: 0.0198634, 0.0204677, 0.0199576, 0.0204610, -0.0005976, 0.0005101

Time for backsubstitution: 2.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0195769, upper bound: 0.0193907
time: 0.36 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0195769, upper bound: 0.0193907
time: 0.35 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0186350, 0.0226922, 0.0210209, 0.0225073, -0.0038722, 0.0016713
1: -0.0215215, -0.0212791, -0.0214985, -0.0212876, -0.0002339, 0.0002194
2: 0.0186700, 0.0191581, 0.0187120, 0.0191457, -0.0004757, 0.0004461
3: -0.0170943, -0.0164003, -0.0169966, -0.0164179, -0.0006764, 0.0005963
4: 0.0198634, 0.0204677, 0.0199614, 0.0204549, -0.0005915, 0.0005062

Time for backsubstitution: 2.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0194648, upper bound: 0.0193907
time: 0.36 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0194648, upper bound: 0.0193907
time: 0.35 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0199707, 0.0226861, 0.0198755, 0.0226901, -0.0027194, 0.0028106
1: -0.0215206, -0.0212867, -0.0215207, -0.0212857, -0.0002349, 0.0002339
2: 0.0186935, 0.0191568, 0.0186918, 0.0191581, -0.0004647, 0.0004650
3: -0.0170405, -0.0164028, -0.0170443, -0.0164011, -0.0006394, 0.0006414
4: 0.0199177, 0.0204661, 0.0199138, 0.0204675, -0.0005498, 0.0005523

Time for backsubstitution: 2.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0196599, upper bound: 0.0196204
time: 0.35 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0196599, upper bound: 0.0196204
time: 0.36 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0199707, 0.0226861, 0.0185295, 0.0226950, -0.0027243, 0.0041566
1: -0.0215206, -0.0212867, -0.0215215, -0.0212783, -0.0002423, 0.0002348
2: 0.0186935, 0.0191568, 0.0186682, 0.0191596, -0.0004662, 0.0004886
3: -0.0170405, -0.0164028, -0.0170986, -0.0163983, -0.0006422, 0.0006957
4: 0.0199177, 0.0204661, 0.0198591, 0.0204693, -0.0005516, 0.0006070

Time for backsubstitution: 2.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0194473, upper bound: 0.0196157
time: 0.36 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0194473, upper bound: 0.0196157
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0186350, 0.0226922, 0.0198755, 0.0226901, -0.0040550, 0.0028167
1: -0.0215215, -0.0212791, -0.0215207, -0.0212857, -0.0002358, 0.0002415
2: 0.0186700, 0.0191581, 0.0186918, 0.0191581, -0.0004881, 0.0004663
3: -0.0170943, -0.0164003, -0.0170443, -0.0164011, -0.0006932, 0.0006440
4: 0.0198634, 0.0204677, 0.0199138, 0.0204675, -0.0006041, 0.0005539

Time for backsubstitution: 2.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0195935, upper bound: 0.0193907
time: 0.36 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0195935, upper bound: 0.0193906
time: 0.35 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0186350, 0.0226922, 0.0185295, 0.0226950, -0.0040599, 0.0041627
1: -0.0215215, -0.0212791, -0.0215215, -0.0212783, -0.0002432, 0.0002424
2: 0.0186700, 0.0191581, 0.0186682, 0.0191596, -0.0004896, 0.0004899
3: -0.0170943, -0.0164003, -0.0170986, -0.0163983, -0.0006960, 0.0006983
4: 0.0198634, 0.0204677, 0.0198591, 0.0204693, -0.0006059, 0.0006086

Time for backsubstitution: 2.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0194241, upper bound: 0.0193907
time: 0.35 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0194241, upper bound: 0.0193906
time: 0.35 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 3.67 seconds
NS_A2_B2_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.67
Output dim: 0, lower bound: -0.0196074, upper bound: 0.0195725
NS_A2_B2_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.67
Output dim: 0, lower bound: -0.0196074, upper bound: 0.0195725
NS_A2_B2_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.67
Output dim: 0, lower bound: -0.0195544, upper bound: 0.0195743
NS_A2_B2_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.67
Output dim: 0, lower bound: -0.0195544, upper bound: 0.0195743
NS_A2_B2_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.67
Output dim: 0, lower bound: -0.0196341, upper bound: 0.0195343
NS_A2_B2_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.67
Output dim: 0, lower bound: -0.0196341, upper bound: 0.0195343
NS_A2_B2_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.67
Output dim: 0, lower bound: -0.0195811, upper bound: 0.0195299
NS_A2_B2_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.67
Output dim: 0, lower bound: -0.0195811, upper bound: 0.0195299
NS_A2_B2_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.67
Output dim: 0, lower bound: -0.0196749, upper bound: 0.0195853
NS_A2_B2_A1_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.67
Output dim: 0, lower bound: -0.0196749, upper bound: 0.0195853
NS_A2_B2_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.67
Output dim: 0, lower bound: -0.0194546, upper bound: 0.0195743
NS_A2_B2_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.67
Output dim: 0, lower bound: -0.0194546, upper bound: 0.0195743
NS_A2_B2_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.67
Output dim: 0, lower bound: -0.0197017, upper bound: 0.0195396
NS_A2_B2_A1_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.67
Output dim: 0, lower bound: -0.0197017, upper bound: 0.0195396
NS_A2_B2_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.67
Output dim: 0, lower bound: -0.0194813, upper bound: 0.0195239
NS_A2_B2_A1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.67
Output dim: 0, lower bound: -0.0194813, upper bound: 0.0195239
NS_A2_B2_A2_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.67
Output dim: 0, lower bound: -0.0196001, upper bound: 0.0195762
NS_A2_B2_A2_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.67
Output dim: 0, lower bound: -0.0196001, upper bound: 0.0195762
NS_A2_B2_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.67
Output dim: 0, lower bound: -0.0195471, upper bound: 0.0195911
NS_A2_B2_A2_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.67
Output dim: 0, lower bound: -0.0195471, upper bound: 0.0195911
NS_A2_B2_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.67
Output dim: 0, lower bound: -0.0195769, upper bound: 0.0193907
NS_A2_B2_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.67
Output dim: 0, lower bound: -0.0195769, upper bound: 0.0193907
NS_A2_B2_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.67
Output dim: 0, lower bound: -0.0194648, upper bound: 0.0193907
NS_A2_B2_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.67
Output dim: 0, lower bound: -0.0194648, upper bound: 0.0193907
NS_A2_B2_A2_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.67
Output dim: 0, lower bound: -0.0196599, upper bound: 0.0196204
NS_A2_B2_A2_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.67
Output dim: 0, lower bound: -0.0196599, upper bound: 0.0196204
NS_A2_B2_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.67
Output dim: 0, lower bound: -0.0194473, upper bound: 0.0196157
NS_A2_B2_A2_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.67
Output dim: 0, lower bound: -0.0194473, upper bound: 0.0196157
NS_A2_B2_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.67
Output dim: 0, lower bound: -0.0195935, upper bound: 0.0193907
NS_A2_B2_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.67
Output dim: 0, lower bound: -0.0195935, upper bound: 0.0193906
NS_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.67
Output dim: 0, lower bound: -0.0194241, upper bound: 0.0193907
NS_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.67
Output dim: 0, lower bound: -0.0194241, upper bound: 0.0193906

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 3.61 + 300.66 = 304.27 seconds
