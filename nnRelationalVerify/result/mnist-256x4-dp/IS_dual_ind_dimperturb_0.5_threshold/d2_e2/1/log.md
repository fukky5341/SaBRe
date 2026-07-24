## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.35483528


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.1972075, 0.2236398, -0.1972075, 0.2236398, -0.4175322, 0.4175321)
1: (-0.0905370, 0.0818782, -0.0905370, 0.0818782, -0.1724152, 0.1724152)
2: (-0.1382934, 0.1247717, -0.1382934, 0.1247717, -0.2630651, 0.2630651)
3: (-0.1110995, 0.1632453, -0.1110995, 0.1632453, -0.2743448, 0.2743448)
4: (-0.0896871, 0.0782065, -0.0896871, 0.0782065, -0.1676241, 0.1676241)
5: (-0.1051460, 0.1293218, -0.1051460, 0.1293218, -0.2333673, 0.2333673)
6: (-0.1137787, 0.1030117, -0.1137787, 0.1030117, -0.2143654, 0.2143653)
7: (-0.0911258, 0.0960266, -0.0911258, 0.0960266, -0.1871524, 0.1871524)
8: (0.4911070, 1.1417875, 0.4911070, 1.1417875, -0.6300468, 0.6300471)
9: (-0.0847257, 0.1675473, -0.0847257, 0.1675473, -0.2384180, 0.2384180)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.18 + 1.73 = 2.91 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.3961252, upper bound: 0.3961252

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3769442, upper bound: 0.3898587
time: 0.87 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3917158, upper bound: 0.3917158
time: 0.91 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.94 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.94
Output dim: 8, lower bound: -0.3769442, upper bound: 0.3898587
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.94
Output dim: 8, lower bound: -0.3917158, upper bound: 0.3917158

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.1838963, 0.1638547, -0.1955043, 0.2128358, -0.3932782, 0.3548834
1: -0.0816319, 0.0725747, -0.0892232, 0.0798731, -0.1615050, 0.1617979
2: -0.1286420, 0.1084636, -0.1371715, 0.1212022, -0.2498442, 0.2456351
3: -0.1052208, 0.1448574, -0.1105539, 0.1590158, -0.2642366, 0.2554113
4: -0.0711292, 0.0696799, -0.0863594, 0.0770241, -0.1477531, 0.1557657
5: -0.0960292, 0.1115803, -0.1039457, 0.1261230, -0.2208555, 0.2155260
6: -0.1061987, 0.0888898, -0.1128734, 0.0986786, -0.2044937, 0.2017632
7: -0.0815241, 0.0849261, -0.0887164, 0.0944807, -0.1760048, 0.1736425
8: 0.5595688, 1.1321422, 0.5047445, 1.1409299, -0.5509927, 0.6037278
9: -0.0721091, 0.1545540, -0.0830181, 0.1649686, -0.2234346, 0.2236810

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3629051, upper bound: 0.3753414
time: 0.75 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3587375, upper bound: 0.3753046
time: 0.88 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.1949567, 0.2096235, -0.1972075, 0.2236398, -0.4152822, 0.4035280
1: -0.0888062, 0.0791439, -0.0905370, 0.0818782, -0.1706844, 0.1696810
2: -0.1368397, 0.1201669, -0.1382934, 0.1247717, -0.2616115, 0.2584603
3: -0.1103789, 0.1567458, -0.1110995, 0.1632453, -0.2736242, 0.2678453
4: -0.0851141, 0.0767242, -0.0896871, 0.0782065, -0.1631211, 0.1661417
5: -0.1035460, 0.1239372, -0.1051460, 0.1293218, -0.2317774, 0.2283401
6: -0.1126414, 0.0971061, -0.1137787, 0.1030117, -0.2132308, 0.2108848
7: -0.0876098, 0.0940467, -0.0911258, 0.0960266, -0.1836364, 0.1851725
8: 0.5102487, 1.1405709, 0.4911070, 1.1417875, -0.6088202, 0.6285541
9: -0.0825494, 0.1635293, -0.0847257, 0.1675473, -0.2362609, 0.2348905

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3898587, upper bound: 0.3769442
time: 0.87 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3898587, upper bound: 0.3917158
time: 0.82 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.88 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.88
Output dim: 8, lower bound: -0.3629051, upper bound: 0.3753414
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.88
Output dim: 8, lower bound: -0.3587375, upper bound: 0.3753046
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.88
Output dim: 8, lower bound: -0.3898587, upper bound: 0.3769442
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.88
Output dim: 8, lower bound: -0.3898587, upper bound: 0.3917158

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.1838963, 0.1638547, -0.1894943, 0.2108713, -0.3912090, 0.3488619
1: -0.0816319, 0.0725747, -0.0861787, 0.0791413, -0.1607732, 0.1587534
2: -0.1286420, 0.1084636, -0.1319686, 0.1185244, -0.2471664, 0.2404322
3: -0.1052208, 0.1448574, -0.1064277, 0.1578168, -0.2630376, 0.2512851
4: -0.0711292, 0.0696799, -0.0850799, 0.0739544, -0.1446831, 0.1543992
5: -0.0960292, 0.1115803, -0.0997375, 0.1249990, -0.2193950, 0.2113178
6: -0.1061987, 0.0888898, -0.1091363, 0.0977056, -0.2018910, 0.1980262
7: -0.0815241, 0.0849261, -0.0880898, 0.0902807, -0.1718048, 0.1730159
8: 0.5595688, 1.1321422, 0.5080640, 1.1336856, -0.5433269, 0.5999939
9: -0.0721091, 0.1545540, -0.0780499, 0.1640685, -0.2220751, 0.2186190

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3587375, upper bound: 0.3753046
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3587375, upper bound: 0.3753046
time: 0.71 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.1785732, 0.1630626, -0.1728526, 0.2369629, -0.4121169, 0.3319895
1: -0.0789322, 0.0706138, -0.0793241, 0.0843088, -0.1632410, 0.1499378
2: -0.1240126, 0.1055070, -0.1166004, 0.1244655, -0.2484781, 0.2221074
3: -0.1015566, 0.1444414, -0.0928212, 0.1686663, -0.2702230, 0.2372625
4: -0.0705063, 0.0669642, -0.0915657, 0.0668451, -0.1369281, 0.1579253
5: -0.0922971, 0.1111453, -0.0882435, 0.1332200, -0.2244238, 0.1984074
6: -0.1028994, 0.0885969, -0.0982061, 0.1083402, -0.2079057, 0.1864284
7: -0.0813158, 0.0812353, -0.0941155, 0.0800154, -0.1613312, 0.1753508
8: 0.5607803, 1.1256993, 0.4745982, 1.1096214, -0.5233302, 0.6330523
9: -0.0677309, 0.1541935, -0.0650607, 0.1699587, -0.2239717, 0.2064286

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3587375, upper bound: 0.3753046
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3587375, upper bound: 0.3753046
time: 0.88 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.1949567, 0.2096235, -0.1838963, 0.1638547, -0.3543363, 0.3902327
1: -0.0888062, 0.0791439, -0.0816319, 0.0725747, -0.1613810, 0.1607758
2: -0.1368397, 0.1201669, -0.1286420, 0.1084636, -0.2453033, 0.2488089
3: -0.1103789, 0.1567458, -0.1052208, 0.1448574, -0.2552363, 0.2619666
4: -0.0851141, 0.0767242, -0.0711292, 0.0696799, -0.1545970, 0.1474531
5: -0.1035460, 0.1239372, -0.0960292, 0.1115803, -0.2151263, 0.2190853
6: -0.1126414, 0.0971061, -0.1061987, 0.0888898, -0.2015312, 0.2033049
7: -0.0876098, 0.0940467, -0.0815241, 0.0849261, -0.1725360, 0.1755708
8: 0.5102487, 1.1405709, 0.5595688, 1.1321422, -0.5978277, 0.5506115
9: -0.0825494, 0.1635293, -0.0721091, 0.1545540, -0.2232112, 0.2223819

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3753414, upper bound: 0.3629051
time: 0.97 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3753046, upper bound: 0.3587375
time: 0.74 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.1949567, 0.2096235, -0.1949567, 0.2096235, -0.4012779, 0.4012780
1: -0.0888062, 0.0791439, -0.0888062, 0.0791439, -0.1679502, 0.1679502
2: -0.1368397, 0.1201669, -0.1368397, 0.1201669, -0.2570066, 0.2570066
3: -0.1103789, 0.1567458, -0.1103789, 0.1567458, -0.2671247, 0.2671247
4: -0.0851141, 0.0767242, -0.0851141, 0.0767242, -0.1616386, 0.1616386
5: -0.1035460, 0.1239372, -0.1035460, 0.1239372, -0.2267502, 0.2267503
6: -0.1126414, 0.0971061, -0.1126414, 0.0971061, -0.2097476, 0.2097476
7: -0.0876098, 0.0940467, -0.0876098, 0.0940467, -0.1816566, 0.1816566
8: 0.5102487, 1.1405709, 0.5102487, 1.1405709, -0.6073270, 0.6073272
9: -0.0825494, 0.1635293, -0.0825494, 0.1635293, -0.2327333, 0.2327333

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3753414, upper bound: 0.3724807
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3753046, upper bound: 0.3709007
time: 0.83 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.88 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.88
Output dim: 8, lower bound: -0.3587375, upper bound: 0.3753046
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.88
Output dim: 8, lower bound: -0.3587375, upper bound: 0.3753046
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.88
Output dim: 8, lower bound: -0.3587375, upper bound: 0.3753046
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.88
Output dim: 8, lower bound: -0.3587375, upper bound: 0.3753046
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.88
Output dim: 8, lower bound: -0.3753414, upper bound: 0.3629051
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.88
Output dim: 8, lower bound: -0.3753046, upper bound: 0.3587375
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.88
Output dim: 8, lower bound: -0.3753414, upper bound: 0.3724807
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.88
Output dim: 8, lower bound: -0.3753046, upper bound: 0.3709007

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1777863, 0.1620202, -0.1894943, 0.2108713, -0.3850802, 0.3469660
1: -0.0784974, 0.0701884, -0.0861787, 0.0791413, -0.1576387, 0.1563672
2: -0.1233589, 0.1049960, -0.1319686, 0.1185244, -0.2418833, 0.2369646
3: -0.1010794, 0.1437785, -0.1064277, 0.1578168, -0.2588963, 0.2502063
4: -0.0700406, 0.0665344, -0.0850799, 0.0739544, -0.1434848, 0.1512517
5: -0.0917463, 0.1104688, -0.0997375, 0.1249990, -0.2150725, 0.2088463
6: -0.1024385, 0.0881734, -0.1091363, 0.0977056, -0.1981291, 0.1965817
7: -0.0809961, 0.0806724, -0.0880898, 0.0902807, -0.1712768, 0.1687621
8: 0.5625700, 1.1248417, 0.5080640, 1.1336856, -0.5399880, 0.5922668
9: -0.0670884, 0.1537427, -0.0780499, 0.1640685, -0.2169523, 0.2173272

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3629051, upper bound: 0.3612224
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3629051, upper bound: 0.3753414
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1614320, 0.1889966, -0.1894943, 0.2108713, -0.3690840, 0.3741715
1: -0.0713606, 0.0757004, -0.0861787, 0.0791413, -0.1505019, 0.1618792
2: -0.1081931, 0.1052305, -0.1319686, 0.1185244, -0.2267175, 0.2371992
3: -0.0879364, 0.1524022, -0.1064277, 0.1578168, -0.2457532, 0.2588299
4: -0.0773016, 0.0592717, -0.0850799, 0.0739544, -0.1506157, 0.1440386
5: -0.0801481, 0.1193002, -0.0997375, 0.1249990, -0.2037790, 0.2166167
6: -0.0918218, 0.0937783, -0.1091363, 0.0977056, -0.1876635, 0.1995748
7: -0.0852482, 0.0704812, -0.0880898, 0.0902807, -0.1755289, 0.1585710
8: 0.5316952, 1.1015277, 0.5080640, 1.1336856, -0.5766463, 0.5722935
9: -0.0542630, 0.1595314, -0.0780499, 0.1640685, -0.2050672, 0.2233371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3629051, upper bound: 0.3612224
time: 1.07 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3629051, upper bound: 0.3753414
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1777863, 0.1620202, -0.1728526, 0.2369629, -0.4112500, 0.3305938
1: -0.0784974, 0.0701884, -0.0793241, 0.0843088, -0.1628062, 0.1495125
2: -0.1233589, 0.1049960, -0.1166004, 0.1244655, -0.2478245, 0.2215964
3: -0.1010794, 0.1437785, -0.0928212, 0.1686663, -0.2697458, 0.2365997
4: -0.0700406, 0.0665344, -0.0915657, 0.0668451, -0.1364128, 0.1574863
5: -0.0917463, 0.1104688, -0.0882435, 0.1332200, -0.2237970, 0.1975379
6: -0.1024385, 0.0881734, -0.0982061, 0.1083402, -0.2074207, 0.1857228
7: -0.0809961, 0.0806724, -0.0941155, 0.0800154, -0.1610116, 0.1747878
8: 0.5625700, 1.1248417, 0.4745982, 1.1096214, -0.5188329, 0.6313243
9: -0.0670884, 0.1537427, -0.0650607, 0.1699587, -0.2230578, 0.2050790

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3587375, upper bound: 0.3590036
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3587375, upper bound: 0.3753046
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1614320, 0.1889966, -0.1728526, 0.2369629, -0.3950732, 0.3576609
1: -0.0713606, 0.0757004, -0.0793241, 0.0843088, -0.1556694, 0.1550245
2: -0.1081931, 0.1052305, -0.1166004, 0.1244655, -0.2326587, 0.2218309
3: -0.0879364, 0.1524022, -0.0928212, 0.1686663, -0.2566027, 0.2452233
4: -0.0773016, 0.0592717, -0.0915657, 0.0668451, -0.1435261, 0.1502514
5: -0.0801481, 0.1193002, -0.0882435, 0.1332200, -0.2123780, 0.2051922
6: -0.0918218, 0.0937783, -0.0982061, 0.1083402, -0.1968993, 0.1886815
7: -0.0852482, 0.0704812, -0.0941155, 0.0800154, -0.1652637, 0.1645967
8: 0.5316952, 1.1015277, 0.4745982, 1.1096214, -0.5544930, 0.6098731
9: -0.0542630, 0.1595314, -0.0650607, 0.1699587, -0.2104897, 0.2104940

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3587375, upper bound: 0.3590036
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3587375, upper bound: 0.3753046
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1889471, 0.2077041, -0.1838963, 0.1638547, -0.3483136, 0.3881946
1: -0.0857553, 0.0777737, -0.0816319, 0.0725747, -0.1583300, 0.1594055
2: -0.1316370, 0.1175050, -0.1286420, 0.1084636, -0.2401006, 0.2461470
3: -0.1062540, 0.1556900, -0.1052208, 0.1448574, -0.2511113, 0.2609108
4: -0.0838112, 0.0736566, -0.0711292, 0.0696799, -0.1531834, 0.1443851
5: -0.0993342, 0.1228417, -0.0960292, 0.1115803, -0.2109145, 0.2174646
6: -0.1089060, 0.0963833, -0.1061987, 0.0888898, -0.1977958, 0.2011264
7: -0.0870833, 0.0898417, -0.0815241, 0.0849261, -0.1720094, 0.1713658
8: 0.5133498, 1.1333268, 0.5595688, 1.1321422, -0.5943015, 0.5429432
9: -0.0775784, 0.1626870, -0.0721091, 0.1545540, -0.2181448, 0.2209195

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3753046, upper bound: 0.3587375
time: 0.92 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3753046, upper bound: 0.3587375
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.1721372, 0.2339249, -0.1785732, 0.1630626, -0.3312742, 0.4091070
1: -0.0787497, 0.0829208, -0.0789322, 0.0706138, -0.1493635, 0.1618531
2: -0.1161624, 0.1203677, -0.1240126, 0.1055070, -0.2216693, 0.2443802
3: -0.0926231, 0.1637407, -0.1015566, 0.1444414, -0.2370645, 0.2652974
4: -0.0901669, 0.0663936, -0.0705063, 0.0669642, -0.1565564, 0.1364766
5: -0.0876733, 0.1310285, -0.0922971, 0.1111453, -0.1978354, 0.2222264
6: -0.0979364, 0.1014621, -0.1028994, 0.0885969, -0.1861615, 0.2010888
7: -0.0910093, 0.0794143, -0.0813158, 0.0812353, -0.1722446, 0.1607301
8: 0.4839187, 1.1091716, 0.5607803, 1.1256993, -0.6234543, 0.5228467
9: -0.0644370, 0.1676409, -0.0677309, 0.1541935, -0.2058071, 0.2217759

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3753046, upper bound: 0.3587375
time: 3.95 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3753046, upper bound: 0.3587375
time: 1.14 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1889471, 0.2077041, -0.1949567, 0.2096235, -0.3952554, 0.3992403
1: -0.0857553, 0.0777737, -0.0888062, 0.0791439, -0.1648992, 0.1665799
2: -0.1316370, 0.1175050, -0.1368397, 0.1201669, -0.2518039, 0.2543448
3: -0.1062540, 0.1556900, -0.1103789, 0.1567458, -0.2629998, 0.2660690
4: -0.0838112, 0.0736566, -0.0851141, 0.0767242, -0.1602250, 0.1585705
5: -0.0993342, 0.1228417, -0.1035460, 0.1239372, -0.2224374, 0.2251297
6: -0.1089060, 0.0963833, -0.1126414, 0.0971061, -0.2060121, 0.2075779
7: -0.0870833, 0.0898417, -0.0876098, 0.0940467, -0.1811300, 0.1774516
8: 0.5133498, 1.1333268, 0.5102487, 1.1405709, -0.6038003, 0.5996592
9: -0.0775784, 0.1626870, -0.0825494, 0.1635293, -0.2276668, 0.2312711

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3797071, upper bound: 0.3709007
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3797071, upper bound: 0.3709007
time: 1.10 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.1721372, 0.2339249, -0.1897497, 0.2086704, -0.3779883, 0.4202931
1: -0.0787497, 0.0829208, -0.0862238, 0.0781484, -0.1568981, 0.1691447
2: -0.1161624, 0.1203677, -0.1323010, 0.1180364, -0.2341987, 0.2526686
3: -0.0926231, 0.1637407, -0.1067381, 0.1562746, -0.2488977, 0.2704789
4: -0.0901669, 0.0663936, -0.0842482, 0.0741206, -0.1637148, 0.1504211
5: -0.0876733, 0.1310285, -0.0999133, 0.1234389, -0.2102010, 0.2299583
6: -0.0979364, 0.1014621, -0.1094186, 0.0967585, -0.1936302, 0.2075985
7: -0.0910093, 0.0794143, -0.0873668, 0.0904775, -0.1814867, 0.1667811
8: 0.4839187, 1.1091716, 0.5116949, 1.1341281, -0.6328740, 0.5792198
9: -0.0644370, 0.1676409, -0.0783156, 0.1630842, -0.2151174, 0.2321849

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3797071, upper bound: 0.3709007
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3797071, upper bound: 0.3709007
time: 0.76 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.72 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.72
Output dim: 8, lower bound: -0.3629051, upper bound: 0.3612224
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.72
Output dim: 8, lower bound: -0.3629051, upper bound: 0.3753414
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.72
Output dim: 8, lower bound: -0.3629051, upper bound: 0.3612224
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.72
Output dim: 8, lower bound: -0.3629051, upper bound: 0.3753414
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.72
Output dim: 8, lower bound: -0.3587375, upper bound: 0.3590036
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.72
Output dim: 8, lower bound: -0.3587375, upper bound: 0.3753046
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.72
Output dim: 8, lower bound: -0.3587375, upper bound: 0.3590036
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.72
Output dim: 8, lower bound: -0.3587375, upper bound: 0.3753046
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.72
Output dim: 8, lower bound: -0.3753046, upper bound: 0.3587375
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.72
Output dim: 8, lower bound: -0.3753046, upper bound: 0.3587375
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.72
Output dim: 8, lower bound: -0.3753046, upper bound: 0.3587375
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.72
Output dim: 8, lower bound: -0.3753046, upper bound: 0.3587375
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.72
Output dim: 8, lower bound: -0.3797071, upper bound: 0.3709007
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.72
Output dim: 8, lower bound: -0.3797071, upper bound: 0.3709007
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.72
Output dim: 8, lower bound: -0.3797071, upper bound: 0.3709007
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.72
Output dim: 8, lower bound: -0.3797071, upper bound: 0.3709007

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1777863, 0.1620202, -0.1777863, 0.1620202, -0.3352598, 0.3352599
1: -0.0784974, 0.0701884, -0.0784974, 0.0701884, -0.1486858, 0.1486858
2: -0.1233589, 0.1049960, -0.1233589, 0.1049960, -0.2283549, 0.2283549
3: -0.1010794, 0.1437785, -0.1010794, 0.1437785, -0.2444966, 0.2444966
4: -0.0700406, 0.0665344, -0.0700406, 0.0665344, -0.1360649, 0.1360649
5: -0.0917463, 0.1104688, -0.0917463, 0.1104688, -0.2007906, 0.2007906
6: -0.1024385, 0.0881734, -0.1024385, 0.0881734, -0.1898897, 0.1898897
7: -0.0809961, 0.0806724, -0.0809961, 0.0806724, -0.1616685, 0.1616685
8: 0.5625700, 1.1248417, 0.5625700, 1.1248417, -0.5300055, 0.5300052
9: -0.0670884, 0.1537427, -0.0670884, 0.1537427, -0.2064279, 0.2064279

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3127464, upper bound: 0.3429854
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3710521, upper bound: 0.3734641
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1777863, 0.1620202, -0.1889471, 0.2077041, -0.3820658, 0.3464177
1: -0.0784974, 0.0701884, -0.0857553, 0.0777737, -0.1562711, 0.1559437
2: -0.1233589, 0.1049960, -0.1316370, 0.1175050, -0.2408639, 0.2366330
3: -0.1010794, 0.1437785, -0.1062540, 0.1556900, -0.2567695, 0.2500325
4: -0.0700406, 0.0665344, -0.0838112, 0.0736566, -0.1431867, 0.1500359
5: -0.0917463, 0.1104688, -0.0993342, 0.1228417, -0.2131421, 0.2084707
6: -0.1024385, 0.0881734, -0.1089060, 0.0963833, -0.1973646, 0.1963523
7: -0.0809961, 0.0806724, -0.0870833, 0.0898417, -0.1708379, 0.1677557
8: 0.5625700, 1.1248417, 0.5133498, 1.1333268, -0.5396047, 0.5865746
9: -0.0670884, 0.1537427, -0.0775784, 0.1626870, -0.2157967, 0.2168531

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3127464, upper bound: 0.3563497
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3710521, upper bound: 0.3838805
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.1614320, 0.1889966, -0.1777863, 0.1620202, -0.3192638, 0.3624654
1: -0.0713606, 0.0757004, -0.0784974, 0.0701884, -0.1415490, 0.1541978
2: -0.1081931, 0.1052305, -0.1233589, 0.1049960, -0.2131891, 0.2285894
3: -0.0879364, 0.1524022, -0.1010794, 0.1437785, -0.2317149, 0.2534816
4: -0.0773016, 0.0592717, -0.0700406, 0.0665344, -0.1431958, 0.1288518
5: -0.0801481, 0.1193002, -0.0917463, 0.1104688, -0.1894971, 0.2085610
6: -0.0918218, 0.0937783, -0.1024385, 0.0881734, -0.1794241, 0.1928828
7: -0.0852482, 0.0704812, -0.0809961, 0.0806724, -0.1659206, 0.1514774
8: 0.5316952, 1.1015277, 0.5625700, 1.1248417, -0.5666633, 0.5100319
9: -0.0542630, 0.1595314, -0.0670884, 0.1537427, -0.1945428, 0.2124377

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3124335, upper bound: 0.3304886
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 59
type: A, layer: 3, pos: 142
type: A, layer: 3, pos: 255
type: A, layer: 3, pos: 177
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 121
type: A, layer: 3, pos: 248
type: A, layer: 3, pos: 180

Time for candidate selection: 4.95 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3494126, upper bound: 0.3457314
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3466999, upper bound: 0.3457314
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.1614320, 0.1889966, -0.1889471, 0.2077041, -0.3660698, 0.3736231
1: -0.0713606, 0.0757004, -0.0857553, 0.0777737, -0.1491342, 0.1614557
2: -0.1081931, 0.1052305, -0.1316370, 0.1175050, -0.2256981, 0.2368675
3: -0.0879364, 0.1524022, -0.1062540, 0.1556900, -0.2436264, 0.2586561
4: -0.0773016, 0.0592717, -0.0838112, 0.0736566, -0.1503177, 0.1428228
5: -0.0801481, 0.1193002, -0.0993342, 0.1228417, -0.2018485, 0.2162410
6: -0.0918218, 0.0937783, -0.1089060, 0.0963833, -0.1868989, 0.1993454
7: -0.0852482, 0.0704812, -0.0870833, 0.0898417, -0.1750900, 0.1575645
8: 0.5316952, 1.1015277, 0.5133498, 1.1333268, -0.5762625, 0.5666010
9: -0.0542630, 0.1595314, -0.0775784, 0.1626870, -0.2039116, 0.2228630

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3124335, upper bound: 0.3499963
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 59
type: A, layer: 3, pos: 142
type: A, layer: 3, pos: 255
type: A, layer: 3, pos: 177
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 121
type: A, layer: 3, pos: 248
type: A, layer: 3, pos: 180

Time for candidate selection: 4.98 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3494126, upper bound: 0.3621466
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3466999, upper bound: 0.3621466
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1777863, 0.1620202, -0.1614320, 0.1889966, -0.3624654, 0.3192639
1: -0.0784974, 0.0701884, -0.0713606, 0.0757004, -0.1541978, 0.1415490
2: -0.1233589, 0.1049960, -0.1081931, 0.1052305, -0.2285894, 0.2131891
3: -0.1010794, 0.1437785, -0.0879364, 0.1524022, -0.2534816, 0.2317149
4: -0.0700406, 0.0665344, -0.0773016, 0.0592717, -0.1288518, 0.1431958
5: -0.0917463, 0.1104688, -0.0801481, 0.1193002, -0.2085610, 0.1894972
6: -0.1024385, 0.0881734, -0.0918218, 0.0937783, -0.1928829, 0.1794241
7: -0.0809961, 0.0806724, -0.0852482, 0.0704812, -0.1514774, 0.1659206
8: 0.5625700, 1.1248417, 0.5316952, 1.1015277, -0.5100319, 0.5666633
9: -0.0670884, 0.1537427, -0.0542630, 0.1595314, -0.2124377, 0.1945428

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2740580, upper bound: 0.3124001
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3594187, upper bound: 0.3622501
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1777863, 0.1620202, -0.1721758, 0.2339249, -0.4082403, 0.3299164
1: -0.0784974, 0.0701884, -0.0787710, 0.0829208, -0.1614182, 0.1489595
2: -0.1233589, 0.1049960, -0.1161624, 0.1221770, -0.2455360, 0.2211584
3: -0.1010794, 0.1437785, -0.0926231, 0.1656919, -0.2667713, 0.2364016
4: -0.0700406, 0.0665344, -0.0901669, 0.0663992, -0.1359669, 0.1561173
5: -0.0917463, 0.1104688, -0.0877033, 0.1310285, -0.2215996, 0.1969960
6: -0.1024385, 0.0881734, -0.0979364, 0.1053677, -0.2044711, 0.1854559
7: -0.0809961, 0.0806724, -0.0924758, 0.0794143, -0.1604105, 0.1731482
8: 0.5625700, 1.1248417, 0.4812696, 1.1091716, -0.5183520, 0.6244435
9: -0.0670884, 0.1537427, -0.0644370, 0.1682963, -0.2214990, 0.2044545

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2740580, upper bound: 0.3300125
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3594187, upper bound: 0.3790129
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.1614320, 0.1889966, -0.1614320, 0.1889966, -0.3462734, 0.3462734
1: -0.0713606, 0.0757004, -0.0713606, 0.0757004, -0.1470610, 0.1470610
2: -0.1081931, 0.1052305, -0.1081931, 0.1052305, -0.2134236, 0.2134236
3: -0.0879364, 0.1524022, -0.0879364, 0.1524022, -0.2403386, 0.2403386
4: -0.0773016, 0.0592717, -0.0773016, 0.0592717, -0.1359573, 0.1359573
5: -0.0801481, 0.1193002, -0.0801481, 0.1193002, -0.1971167, 0.1971167
6: -0.0918218, 0.0937783, -0.0918218, 0.0937783, -0.1823429, 0.1823429
7: -0.0852482, 0.0704812, -0.0852482, 0.0704812, -0.1557294, 0.1557294
8: 0.5316952, 1.1015277, 0.5316952, 1.1015277, -0.5453138, 0.5453138
9: -0.0542630, 0.1595314, -0.0542630, 0.1595314, -0.1998280, 0.1998280

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 59
type: A, layer: 3, pos: 142
type: A, layer: 3, pos: 255
type: A, layer: 3, pos: 177
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 121
type: A, layer: 3, pos: 248
type: A, layer: 3, pos: 180

Time for candidate selection: 4.18 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3445386, upper bound: 0.3428190
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3428189
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.1614320, 0.1889966, -0.1721758, 0.2339249, -0.3920780, 0.3569839
1: -0.0713606, 0.0757004, -0.0787710, 0.0829208, -0.1542814, 0.1544715
2: -0.1081931, 0.1052305, -0.1161624, 0.1221770, -0.2303702, 0.2213929
3: -0.0879364, 0.1524022, -0.0926231, 0.1656919, -0.2536283, 0.2450253
4: -0.0773016, 0.0592717, -0.0901669, 0.0663992, -0.1430801, 0.1488829
5: -0.0801481, 0.1193002, -0.0877033, 0.1310285, -0.2101901, 0.2046513
6: -0.0918218, 0.0937783, -0.0979364, 0.1053677, -0.1939607, 0.1884146
7: -0.0852482, 0.0704812, -0.0924758, 0.0794143, -0.1646625, 0.1629571
8: 0.5316952, 1.1015277, 0.4812696, 1.1091716, -0.5540092, 0.6032212
9: -0.0542630, 0.1595314, -0.0644370, 0.1682963, -0.2089555, 0.2098725

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 59
type: A, layer: 3, pos: 142
type: A, layer: 3, pos: 255
type: A, layer: 3, pos: 177
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 121
type: A, layer: 3, pos: 248
type: A, layer: 3, pos: 180

Time for candidate selection: 4.10 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3445386, upper bound: 0.3618980
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3618980
time: 1.01 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1889471, 0.2077041, -0.1777863, 0.1620202, -0.3464175, 0.3820658
1: -0.0857553, 0.0777737, -0.0784974, 0.0701884, -0.1559437, 0.1562711
2: -0.1316370, 0.1175050, -0.1233589, 0.1049960, -0.2366330, 0.2408639
3: -0.1062540, 0.1556900, -0.1010794, 0.1437785, -0.2500325, 0.2567695
4: -0.0838112, 0.0736566, -0.0700406, 0.0665344, -0.1500359, 0.1431868
5: -0.0993342, 0.1228417, -0.0917463, 0.1104688, -0.2084707, 0.2131420
6: -0.1089060, 0.0963833, -0.1024385, 0.0881734, -0.1963523, 0.1973646
7: -0.0870833, 0.0898417, -0.0809961, 0.0806724, -0.1677557, 0.1708379
8: 0.5133498, 1.1333268, 0.5625700, 1.1248417, -0.5865746, 0.5396044
9: -0.0775784, 0.1626870, -0.0670884, 0.1537427, -0.2168531, 0.2157967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2833296, upper bound: 0.3124163
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3740909, upper bound: 0.3614046
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1889471, 0.2077041, -0.1614320, 0.1889966, -0.3736231, 0.3660698
1: -0.0857553, 0.0777737, -0.0713606, 0.0757004, -0.1614557, 0.1491342
2: -0.1316370, 0.1175050, -0.1081931, 0.1052305, -0.2368675, 0.2256981
3: -0.1062540, 0.1556900, -0.0879364, 0.1524022, -0.2586561, 0.2436264
4: -0.0838112, 0.0736566, -0.0773016, 0.0592717, -0.1428228, 0.1503177
5: -0.0993342, 0.1228417, -0.0801481, 0.1193002, -0.2162410, 0.2018486
6: -0.1089060, 0.0963833, -0.0918218, 0.0937783, -0.1993454, 0.1868989
7: -0.0870833, 0.0898417, -0.0852482, 0.0704812, -0.1575645, 0.1750900
8: 0.5133498, 1.1333268, 0.5316952, 1.1015277, -0.5666010, 0.5762625
9: -0.0775784, 0.1626870, -0.0542630, 0.1595314, -0.2228630, 0.2039116

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2833296, upper bound: 0.3124163
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3740909, upper bound: 0.3614046
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.1721372, 0.2339249, -0.1777863, 0.1620202, -0.3298780, 0.4082403
1: -0.0787497, 0.0829208, -0.0784974, 0.0701884, -0.1489381, 0.1614182
2: -0.1161624, 0.1203677, -0.1233589, 0.1049960, -0.2211584, 0.2437266
3: -0.0926231, 0.1637407, -0.1010794, 0.1437785, -0.2364016, 0.2648202
4: -0.0901669, 0.0663936, -0.0700406, 0.0665344, -0.1561173, 0.1359613
5: -0.0876733, 0.1310285, -0.0917463, 0.1104688, -0.1969641, 0.2215996
6: -0.0979364, 0.1014621, -0.1024385, 0.0881734, -0.1854559, 0.2006039
7: -0.0910093, 0.0794143, -0.0809961, 0.0806724, -0.1716817, 0.1604105
8: 0.4839187, 1.1091716, 0.5625700, 1.1248417, -0.6217270, 0.5183520
9: -0.0644370, 0.1676409, -0.0670884, 0.1537427, -0.2044545, 0.2208621

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 59
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 142
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 255
type: A, layer: 3, pos: 140
type: A, layer: 3, pos: 177
type: A, layer: 3, pos: 248
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 121
type: A, layer: 3, pos: 91
type: A, layer: 3, pos: 180

Time for candidate selection: 4.20 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3584130, upper bound: 0.3427173
time: 1.00 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3618979, upper bound: 0.3427173
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.1721372, 0.2339249, -0.1614320, 0.1889966, -0.3569455, 0.3920779
1: -0.0787497, 0.0829208, -0.0713606, 0.0757004, -0.1544501, 0.1542814
2: -0.1161624, 0.1203677, -0.1081931, 0.1052305, -0.2213929, 0.2285608
3: -0.0926231, 0.1637407, -0.0879364, 0.1524022, -0.2450253, 0.2516771
4: -0.0901669, 0.0663936, -0.0773016, 0.0592717, -0.1488829, 0.1430746
5: -0.0876733, 0.1310285, -0.0801481, 0.1193002, -0.2046204, 0.2101901
6: -0.0979364, 0.1014621, -0.0918218, 0.0937783, -0.1884147, 0.1900784
7: -0.0910093, 0.0794143, -0.0852482, 0.0704812, -0.1614905, 0.1646625
8: 0.4839187, 1.1091716, 0.5316952, 1.1015277, -0.6004977, 0.5540092
9: -0.0644370, 0.1676409, -0.0542630, 0.1595314, -0.2098725, 0.2083152

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 59
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 142
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 255
type: A, layer: 3, pos: 140
type: A, layer: 3, pos: 177
type: A, layer: 3, pos: 248
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 121
type: A, layer: 3, pos: 91
type: A, layer: 3, pos: 180

Time for candidate selection: 4.12 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3584130, upper bound: 0.3427173
time: 1.02 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3618979, upper bound: 0.3427173
time: 1.34 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1889471, 0.2077041, -0.1889471, 0.2077041, -0.3932178, 0.3932178
1: -0.0857553, 0.0777737, -0.0857553, 0.0777737, -0.1635289, 0.1635289
2: -0.1316370, 0.1175050, -0.1316370, 0.1175050, -0.2491420, 0.2491420
3: -0.1062540, 0.1556900, -0.1062540, 0.1556900, -0.2619440, 0.2619440
4: -0.0838112, 0.0736566, -0.0838112, 0.0736566, -0.1571569, 0.1571569
5: -0.0993342, 0.1228417, -0.0993342, 0.1228417, -0.2208169, 0.2208169
6: -0.1089060, 0.0963833, -0.1089060, 0.0963833, -0.2038244, 0.2038244
7: -0.0870833, 0.0898417, -0.0870833, 0.0898417, -0.1769250, 0.1769250
8: 0.5133498, 1.1333268, 0.5133498, 1.1333268, -0.5961323, 0.5961325
9: -0.0775784, 0.1626870, -0.0775784, 0.1626870, -0.2262046, 0.2262046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2925581, upper bound: 0.3300583
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3784539, upper bound: 0.3711575
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1889471, 0.2077041, -0.1721372, 0.2339249, -0.4193926, 0.3766787
1: -0.0857553, 0.0777737, -0.0787497, 0.0829208, -0.1686761, 0.1565233
2: -0.1316370, 0.1175050, -0.1161624, 0.1203677, -0.2520047, 0.2336674
3: -0.1062540, 0.1556900, -0.0926231, 0.1637407, -0.2699947, 0.2483131
4: -0.0838112, 0.0736566, -0.0901669, 0.0663936, -0.1499315, 0.1632384
5: -0.0993342, 0.1228417, -0.0876733, 0.1310285, -0.2292749, 0.2093105
6: -0.1089060, 0.0963833, -0.0979364, 0.1014621, -0.2070630, 0.1929282
7: -0.0870833, 0.0898417, -0.0910093, 0.0794143, -0.1664976, 0.1808510
8: 0.5133498, 1.1333268, 0.4839187, 1.1091716, -0.5748801, 0.6312835
9: -0.0775784, 0.1626870, -0.0644370, 0.1676409, -0.2312720, 0.2138064

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2925581, upper bound: 0.3300583
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3784539, upper bound: 0.3711575
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.1721372, 0.2339249, -0.1889471, 0.2077041, -0.3766787, 0.4193926
1: -0.0787497, 0.0829208, -0.0857553, 0.0777737, -0.1565233, 0.1686761
2: -0.1161624, 0.1203677, -0.1316370, 0.1175050, -0.2336674, 0.2520047
3: -0.0926231, 0.1637407, -0.1062540, 0.1556900, -0.2483131, 0.2699947
4: -0.0901669, 0.0663936, -0.0838112, 0.0736566, -0.1632384, 0.1499314
5: -0.0876733, 0.1310285, -0.0993342, 0.1228417, -0.2093106, 0.2292748
6: -0.0979364, 0.1014621, -0.1089060, 0.0963833, -0.1929282, 0.2070630
7: -0.0910093, 0.0794143, -0.0870833, 0.0898417, -0.1808510, 0.1664976
8: 0.4839187, 1.1091716, 0.5133498, 1.1333268, -0.6312838, 0.5748804
9: -0.0644370, 0.1676409, -0.0775784, 0.1626870, -0.2138064, 0.2312720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3120100, upper bound: 0.3394662
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2952244, upper bound: 0.2913433
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.1721372, 0.2339249, -0.1721372, 0.2339249, -0.4027442, 0.4027442
1: -0.0787497, 0.0829208, -0.0787497, 0.0829208, -0.1616705, 0.1616705
2: -0.1161624, 0.1203677, -0.1161624, 0.1203677, -0.2365300, 0.2365300
3: -0.0926231, 0.1637407, -0.0926231, 0.1637407, -0.2563638, 0.2563638
4: -0.0901669, 0.0663936, -0.0901669, 0.0663936, -0.1559994, 0.1559994
5: -0.0876733, 0.1310285, -0.0876733, 0.1310285, -0.2176886, 0.2176886
6: -0.0979364, 0.1014621, -0.0979364, 0.1014621, -0.1961467, 0.1961467
7: -0.0910093, 0.0794143, -0.0910093, 0.0794143, -0.1704236, 0.1704236
8: 0.4839187, 1.1091716, 0.4839187, 1.1091716, -0.6091497, 0.6091495
9: -0.0644370, 0.1676409, -0.0644370, 0.1676409, -0.2183439, 0.2183439

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3120100, upper bound: 0.3394662
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2952244, upper bound: 0.2913433
time: 0.60 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.44 seconds
IS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.44
Output dim: 8, lower bound: -0.3127464, upper bound: 0.3429854
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.44
Output dim: 8, lower bound: -0.3710521, upper bound: 0.3734641
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.44
Output dim: 8, lower bound: -0.3127464, upper bound: 0.3563497
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.44
Output dim: 8, lower bound: -0.3710521, upper bound: 0.3838805
IS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.44
Output dim: 8, lower bound: -0.3494126, upper bound: 0.3457314
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.44
Output dim: 8, lower bound: -0.3466999, upper bound: 0.3457314
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.44
Output dim: 8, lower bound: -0.3494126, upper bound: 0.3621466
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.44
Output dim: 8, lower bound: -0.3466999, upper bound: 0.3621466
IS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.44
Output dim: 8, lower bound: -0.2740580, upper bound: 0.3124001
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.44
Output dim: 8, lower bound: -0.3594187, upper bound: 0.3622501
IS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.44
Output dim: 8, lower bound: -0.2740580, upper bound: 0.3300125
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.44
Output dim: 8, lower bound: -0.3594187, upper bound: 0.3790129
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.44
Output dim: 8, lower bound: -0.3445386, upper bound: 0.3428190
IS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.44
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3428189
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.44
Output dim: 8, lower bound: -0.3445386, upper bound: 0.3618980
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.44
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3618980
IS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.44
Output dim: 8, lower bound: -0.2833296, upper bound: 0.3124163
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.44
Output dim: 8, lower bound: -0.3740909, upper bound: 0.3614046
IS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.44
Output dim: 8, lower bound: -0.2833296, upper bound: 0.3124163
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.44
Output dim: 8, lower bound: -0.3740909, upper bound: 0.3614046
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.44
Output dim: 8, lower bound: -0.3584130, upper bound: 0.3427173
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.44
Output dim: 8, lower bound: -0.3618979, upper bound: 0.3427173
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.44
Output dim: 8, lower bound: -0.3584130, upper bound: 0.3427173
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.44
Output dim: 8, lower bound: -0.3618979, upper bound: 0.3427173
IS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.44
Output dim: 8, lower bound: -0.2925581, upper bound: 0.3300583
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.44
Output dim: 8, lower bound: -0.3784539, upper bound: 0.3711575
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.44
Output dim: 8, lower bound: -0.2925581, upper bound: 0.3300583
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.44
Output dim: 8, lower bound: -0.3784539, upper bound: 0.3711575
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.44
Output dim: 8, lower bound: -0.3120100, upper bound: 0.3394662
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.44
Output dim: 8, lower bound: -0.2952244, upper bound: 0.2913433
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.44
Output dim: 8, lower bound: -0.3120100, upper bound: 0.3394662
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.44
Output dim: 8, lower bound: -0.2952244, upper bound: 0.2913433

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1775892, 0.1604986, -0.1777863, 0.1620202, -0.3350624, 0.3333635
1: -0.0783708, 0.0696102, -0.0784974, 0.0701884, -0.1485593, 0.1481076
2: -0.1232215, 0.1047996, -0.1233589, 0.1049960, -0.2282175, 0.2281586
3: -0.1010032, 0.1423073, -0.1010794, 0.1437785, -0.2444066, 0.2426906
4: -0.0692312, 0.0664058, -0.0700406, 0.0665344, -0.1352481, 0.1359364
5: -0.0916116, 0.1089798, -0.0917463, 0.1104688, -0.2006561, 0.1999088
6: -0.1023371, 0.0872673, -0.1024385, 0.0881734, -0.1897884, 0.1895859
7: -0.0803046, 0.0805327, -0.0809961, 0.0806724, -0.1609770, 0.1615288
8: 0.5658924, 1.1247110, 0.5625700, 1.1248417, -0.5233030, 0.5298686
9: -0.0669297, 0.1527999, -0.0670884, 0.1537427, -0.2062640, 0.2045898

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3429854, upper bound: 0.3129795
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3429854, upper bound: 0.3129795
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1899755, 0.1435920, -0.1884007, 0.2038944, -0.3906219, 0.3286326
1: -0.0847272, 0.0756130, -0.0854059, 0.0760684, -0.1607956, 0.1610189
2: -0.1342172, 0.1120768, -0.1312814, 0.1163454, -0.2505627, 0.2433582
3: -0.1098307, 0.1198920, -0.1060762, 0.1518788, -0.2617095, 0.2259682
4: -0.0672577, 0.0727668, -0.0820632, 0.0733099, -0.1405677, 0.1546340
5: -0.1002874, 0.1072509, -0.0989575, 0.1189843, -0.2191195, 0.2062084
6: -0.1103239, 0.0891246, -0.1086660, 0.0940390, -0.2043628, 0.1977906
7: -0.0697513, 0.0895727, -0.0852913, 0.0894951, -0.1592465, 0.1748641
8: 0.6093661, 1.1399536, 0.5220076, 1.1329693, -0.4932699, 0.5934985
9: -0.0776297, 0.1388457, -0.0771838, 0.1602682, -0.2241305, 0.2108287

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3094543, upper bound: 0.3202145
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3094543, upper bound: 0.3563497
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1775892, 0.1604986, -0.1889471, 0.2077041, -0.3818684, 0.3445349
1: -0.0783708, 0.0696102, -0.0857553, 0.0777737, -0.1561445, 0.1553655
2: -0.1232215, 0.1047996, -0.1316370, 0.1175050, -0.2407265, 0.2364366
3: -0.1010032, 0.1423073, -0.1062540, 0.1556900, -0.2566933, 0.2485612
4: -0.0692312, 0.0664058, -0.0838112, 0.0736566, -0.1423719, 0.1499073
5: -0.0916116, 0.1089798, -0.0993342, 0.1228417, -0.2130075, 0.2075846
6: -0.1023371, 0.0872673, -0.1089060, 0.0963833, -0.1972632, 0.1960721
7: -0.0803046, 0.0805327, -0.0870833, 0.0898417, -0.1701463, 0.1676160
8: 0.5658924, 1.1247110, 0.5133498, 1.1333268, -0.5325313, 0.5864379
9: -0.0669297, 0.1527999, -0.0775784, 0.1626870, -0.2156329, 0.2148892

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3426430, upper bound: 0.3247033
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3426430, upper bound: 0.3838805
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1574272, 0.1647350, -0.1889471, 0.2077041, -0.3617723, 0.3491917
1: -0.0686271, 0.0689221, -0.0857553, 0.0777737, -0.1464007, 0.1546774
2: -0.1053974, 0.0966751, -0.1316370, 0.1175050, -0.2229024, 0.2283121
3: -0.0864605, 0.1418168, -0.1062540, 0.1556900, -0.2421505, 0.2480707
4: -0.0690144, 0.0566287, -0.0838112, 0.0736566, -0.1420149, 0.1401393
5: -0.0774258, 0.1084655, -0.0993342, 0.1228417, -0.1988800, 0.2049201
6: -0.0898071, 0.0869272, -0.1089060, 0.0963833, -0.1847015, 0.1924525
7: -0.0800737, 0.0672017, -0.0870833, 0.0898417, -0.1699155, 0.1542850
8: 0.5648291, 1.0987828, 0.5133498, 1.1333268, -0.5393224, 0.5613370
9: -0.0507631, 0.1523322, -0.0775784, 0.1626870, -0.1996116, 0.2152625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2994156, upper bound: 0.2697806
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3479309, upper bound: 0.3609453
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.1600223, 0.1761911, -0.1889471, 0.2077041, -0.3646598, 0.3604429
1: -0.0703633, 0.0695929, -0.0857553, 0.0777737, -0.1481369, 0.1553482
2: -0.1073237, 0.0999089, -0.1316370, 0.1175050, -0.2248287, 0.2315459
3: -0.0875995, 0.1429007, -0.1062540, 0.1556900, -0.2432895, 0.2491547
4: -0.0714792, 0.0582858, -0.0838112, 0.0736566, -0.1444641, 0.1418366
5: -0.0791898, 0.1096610, -0.0993342, 0.1228417, -0.2008868, 0.2063878
6: -0.0912108, 0.0878678, -0.1089060, 0.0963833, -0.1862936, 0.1933434
7: -0.0807419, 0.0691984, -0.0870833, 0.0898417, -0.1705836, 0.1562817
8: 0.5550525, 1.1008596, 0.5133498, 1.1333268, -0.5496345, 0.5659051
9: -0.0531513, 0.1534001, -0.0775784, 0.1626870, -0.2027828, 0.2159452

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2934464, upper bound: 0.2685264
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3451735, upper bound: 0.3609453
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1775892, 0.1604986, -0.1614320, 0.1889966, -0.3622680, 0.3174293
1: -0.0783708, 0.0696102, -0.0713606, 0.0757004, -0.1540713, 0.1409708
2: -0.1232215, 0.1047996, -0.1081931, 0.1052305, -0.2284521, 0.2129927
3: -0.1010032, 0.1423073, -0.0879364, 0.1524022, -0.2534054, 0.2302127
4: -0.0692312, 0.0664058, -0.0773016, 0.0592717, -0.1280432, 0.1430673
5: -0.0916116, 0.1089798, -0.0801481, 0.1193002, -0.2084264, 0.1886888
6: -0.1023371, 0.0872673, -0.0918218, 0.0937783, -0.1927815, 0.1790891
7: -0.0803046, 0.0805327, -0.0852482, 0.0704812, -0.1507858, 0.1657809
8: 0.5658924, 1.1247110, 0.5316952, 1.1015277, -0.5036469, 0.5665267
9: -0.0669297, 0.1527999, -0.0542630, 0.1595314, -0.2122739, 0.1928638

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3290507, upper bound: 0.3109484
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 59
type: B, layer: 3, pos: 142
type: B, layer: 3, pos: 255
type: B, layer: 3, pos: 177
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 121
type: B, layer: 3, pos: 248
type: B, layer: 3, pos: 180

Time for candidate selection: 4.84 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3441772, upper bound: 0.3486989
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3441772, upper bound: 0.3459820
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1775892, 0.1604986, -0.1721758, 0.2339249, -0.4080429, 0.3280385
1: -0.0783708, 0.0696102, -0.0787710, 0.0829208, -0.1612917, 0.1483812
2: -0.1232215, 0.1047996, -0.1161624, 0.1221770, -0.2453986, 0.2209620
3: -0.1010032, 0.1423073, -0.0926231, 0.1656919, -0.2666951, 0.2349304
4: -0.0692312, 0.0664058, -0.0901669, 0.0663992, -0.1351525, 0.1559888
5: -0.0916116, 0.1089798, -0.0877033, 0.1310285, -0.2214651, 0.1961392
6: -0.1023371, 0.0872673, -0.0979364, 0.1053677, -0.2043698, 0.1851777
7: -0.0803046, 0.0805327, -0.0924758, 0.0794143, -0.1597189, 0.1730085
8: 0.5658924, 1.1247110, 0.4812696, 1.1091716, -0.5116811, 0.6243067
9: -0.0669297, 0.1527999, -0.0644370, 0.1682963, -0.2213352, 0.2026246

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3290531, upper bound: 0.3281036
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2725724, upper bound: 0.3113114
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1574272, 0.1647350, -0.1721464, 0.2337104, -0.3876227, 0.3325406
1: -0.0686271, 0.0689221, -0.0787485, 0.0828608, -0.1514879, 0.1476706
2: -0.1053974, 0.0966751, -0.1161438, 0.1220632, -0.2274606, 0.2128189
3: -0.0864605, 0.1418168, -0.0926156, 0.1655632, -0.2520237, 0.2344323
4: -0.0690144, 0.0566287, -0.0900941, 0.0663777, -0.1347569, 0.1461350
5: -0.0774258, 0.1084655, -0.0876820, 0.1309329, -0.2071675, 0.1933051
6: -0.0898071, 0.0869272, -0.0979222, 0.1052378, -0.1916739, 0.1815193
7: -0.0800737, 0.0672017, -0.0924041, 0.0793876, -0.1594613, 0.1596057
8: 0.5648291, 1.0987828, 0.4816074, 1.1091582, -0.5173426, 0.5980630
9: -0.0507631, 0.1523322, -0.0644103, 0.1682228, -0.2047799, 0.2024082

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 59
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 142
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 255
type: B, layer: 3, pos: 140
type: B, layer: 3, pos: 177
type: B, layer: 3, pos: 248
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 121
type: B, layer: 3, pos: 91
type: B, layer: 3, pos: 180

Time for candidate selection: 4.20 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3584130
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3618980
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.1600223, 0.1761911, -0.1721758, 0.2339249, -0.3906678, 0.3437271
1: -0.0703633, 0.0695929, -0.0787710, 0.0829208, -0.1532841, 0.1483639
2: -0.1073237, 0.0999089, -0.1161624, 0.1221770, -0.2295008, 0.2160712
3: -0.0875995, 0.1429007, -0.0926231, 0.1656919, -0.2532914, 0.2355238
4: -0.0714792, 0.0582858, -0.0901669, 0.0663992, -0.1372153, 0.1478967
5: -0.0791898, 0.1096610, -0.0877033, 0.1310285, -0.2092294, 0.1947005
6: -0.0912108, 0.0878678, -0.0979364, 0.1053677, -0.1933546, 0.1823682
7: -0.0807419, 0.0691984, -0.0924758, 0.0794143, -0.1601562, 0.1616743
8: 0.5550525, 1.1008596, 0.4812696, 1.1091716, -0.5268333, 0.6025259
9: -0.0531513, 0.1534001, -0.0644370, 0.1682963, -0.2078449, 0.2028306

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 59
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 142
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 255
type: B, layer: 3, pos: 140
type: B, layer: 3, pos: 177
type: B, layer: 3, pos: 248
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 121
type: B, layer: 3, pos: 91
type: B, layer: 3, pos: 180

Time for candidate selection: 4.27 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3584130
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3618980
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1886644, 0.2059052, -0.1777863, 0.1620202, -0.3461343, 0.3799293
1: -0.0855726, 0.0766896, -0.0784974, 0.0701884, -0.1557610, 0.1551870
2: -0.1314478, 0.1169224, -0.1233589, 0.1049960, -0.2364438, 0.2402813
3: -0.1061568, 0.1538208, -0.1010794, 0.1437785, -0.2499353, 0.2549003
4: -0.0829563, 0.0734682, -0.0700406, 0.0665344, -0.1491802, 0.1429982
5: -0.0991381, 0.1209499, -0.0917463, 0.1104688, -0.2083155, 0.2112726
6: -0.1087681, 0.0952346, -0.1024385, 0.0881734, -0.1962143, 0.1968237
7: -0.0862061, 0.0896484, -0.0809961, 0.0806724, -0.1668785, 0.1706446
8: 0.5175176, 1.1331558, 0.5625700, 1.1248417, -0.5791759, 0.5394318
9: -0.0773586, 0.1615027, -0.0670884, 0.1537427, -0.2166286, 0.2138029

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3563497, upper bound: 0.3127464
time: 0.90 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3563497, upper bound: 0.3710521
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1886644, 0.2059052, -0.1614320, 0.1889966, -0.3733399, 0.3639951
1: -0.0855726, 0.0766896, -0.0713606, 0.0757004, -0.1612730, 0.1480502
2: -0.1314478, 0.1169224, -0.1081931, 0.1052305, -0.2366783, 0.2251155
3: -0.1061568, 0.1538208, -0.0879364, 0.1524022, -0.2585590, 0.2417572
4: -0.0829563, 0.0734682, -0.0773016, 0.0592717, -0.1419753, 0.1501291
5: -0.0991381, 0.1209499, -0.0801481, 0.1193002, -0.2160858, 0.2000526
6: -0.1087681, 0.0952346, -0.0918218, 0.0937783, -0.1992074, 0.1863705
7: -0.0862061, 0.0896484, -0.0852482, 0.0704812, -0.1566873, 0.1748966
8: 0.5175176, 1.1331558, 0.5316952, 1.1015277, -0.5595202, 0.5760899
9: -0.0773586, 0.1615027, -0.0542630, 0.1595314, -0.2226385, 0.2020769

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3486403, upper bound: 0.3113746
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 59
type: B, layer: 3, pos: 142
type: B, layer: 3, pos: 255
type: B, layer: 3, pos: 177
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 121
type: B, layer: 3, pos: 248
type: B, layer: 3, pos: 180

Time for candidate selection: 4.87 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3609453, upper bound: 0.3479309
time: 1.34 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3609453, upper bound: 0.3451735
time: 1.85 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1660702, 0.2018342, -0.1777863, 0.1620202, -0.3235768, 0.3755190
1: -0.0747557, 0.0741434, -0.0784974, 0.0701884, -0.1449441, 0.1526408
2: -0.1117253, 0.1088224, -0.1233589, 0.1049960, -0.2167213, 0.2321813
3: -0.0900854, 0.1500339, -0.1010794, 0.1437785, -0.2338639, 0.2511133
4: -0.0792971, 0.0625792, -0.0700406, 0.0665344, -0.1452653, 0.1321144
5: -0.0834441, 0.1170112, -0.0917463, 0.1104688, -0.1925472, 0.2068741
6: -0.0948899, 0.0926307, -0.1024385, 0.0881734, -0.1822849, 0.1917270
7: -0.0843168, 0.0746323, -0.0809961, 0.0806724, -0.1649892, 0.1556284
8: 0.5274178, 1.1043098, 0.5625700, 1.1248417, -0.5730352, 0.5115328
9: -0.0594005, 0.1584760, -0.0670884, 0.1537427, -0.1986213, 0.2111707

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3146726, upper bound: 0.2602647
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3628378, upper bound: 0.3439668
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.1708696, 0.2229829, -0.1777863, 0.1620202, -0.3286098, 0.3969949
1: -0.0778513, 0.0773110, -0.0784974, 0.0701884, -0.1480397, 0.1558084
2: -0.1154248, 0.1155212, -0.1233589, 0.1049960, -0.2204208, 0.2388801
3: -0.0923715, 0.1550202, -0.1010794, 0.1437785, -0.2361501, 0.2560996
4: -0.0852941, 0.0655414, -0.0700406, 0.0665344, -0.1513329, 0.1351088
5: -0.0867782, 0.1221924, -0.0917463, 0.1104688, -0.1961401, 0.2125453
6: -0.0974704, 0.0960724, -0.1024385, 0.0881734, -0.1850003, 0.1951326
7: -0.0868933, 0.0782940, -0.0809961, 0.0806724, -0.1675657, 0.1592901
8: 0.5048499, 1.1085424, 0.5625700, 1.1248417, -0.5977392, 0.5177584
9: -0.0635459, 0.1620800, -0.0670884, 0.1537427, -0.2035277, 0.2144784

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3151583, upper bound: 0.2600348
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3653816, upper bound: 0.3439668
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1660702, 0.2018342, -0.1614320, 0.1889966, -0.3506675, 0.3593719
1: -0.0747557, 0.0741434, -0.0713606, 0.0757004, -0.1504562, 0.1455040
2: -0.1117253, 0.1088224, -0.1081931, 0.1052305, -0.2169558, 0.2170155
3: -0.0900854, 0.1500339, -0.0879364, 0.1524022, -0.2424875, 0.2379702
4: -0.0792971, 0.0625792, -0.0773016, 0.0592717, -0.1380313, 0.1392310
5: -0.0834441, 0.1170112, -0.0801481, 0.1193002, -0.2002179, 0.1954561
6: -0.0948899, 0.0926307, -0.0918218, 0.0937783, -0.1852549, 0.1811957
7: -0.0843168, 0.0746323, -0.0852482, 0.0704812, -0.1547980, 0.1598805
8: 0.5274178, 1.1043098, 0.5316952, 1.1015277, -0.5519900, 0.5473487
9: -0.0594005, 0.1584760, -0.0542630, 0.1595314, -0.2041948, 0.1987075

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3034707, upper bound: 0.2393176
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 59
type: B, layer: 3, pos: 142
type: B, layer: 3, pos: 255
type: B, layer: 3, pos: 177
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 121
type: B, layer: 3, pos: 248
type: B, layer: 3, pos: 180

Time for candidate selection: 4.75 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3584130, upper bound: 0.3427173
time: 0.92 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3584130, upper bound: 0.3427173
time: 2.45 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.1708696, 0.2229829, -0.1614320, 0.1889966, -0.3556769, 0.3806784
1: -0.0778513, 0.0773110, -0.0713606, 0.0757004, -0.1535518, 0.1486716
2: -0.1154248, 0.1155212, -0.1081931, 0.1052305, -0.2206554, 0.2237143
3: -0.0923715, 0.1550202, -0.0879364, 0.1524022, -0.2447737, 0.2429566
4: -0.0852941, 0.0655414, -0.0773016, 0.0592717, -0.1440748, 0.1422220
5: -0.0867782, 0.1221924, -0.0801481, 0.1193002, -0.2037984, 0.2009923
6: -0.0974704, 0.0960724, -0.0918218, 0.0937783, -0.1879591, 0.1845353
7: -0.0868933, 0.0782940, -0.0852482, 0.0704812, -0.1573745, 0.1635422
8: 0.5048499, 1.1085424, 0.5316952, 1.1015277, -0.5754800, 0.5534177
9: -0.0635459, 0.1620800, -0.0542630, 0.1595314, -0.2089722, 0.2016861

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 59
type: B, layer: 3, pos: 142
type: B, layer: 3, pos: 255
type: B, layer: 3, pos: 177
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 121
type: B, layer: 3, pos: 248
type: B, layer: 3, pos: 180

Time for candidate selection: 4.26 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3618980, upper bound: 0.3427173
time: 0.90 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3618980, upper bound: 0.3427173
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1886644, 0.2059052, -0.1889471, 0.2077041, -0.3929341, 0.3910954
1: -0.0855726, 0.0766896, -0.0857553, 0.0777737, -0.1633462, 0.1624449
2: -0.1314478, 0.1169224, -0.1316370, 0.1175050, -0.2489528, 0.2485594
3: -0.1061568, 0.1538208, -0.1062540, 0.1556900, -0.2618468, 0.2600748
4: -0.0829563, 0.0734682, -0.0838112, 0.0736566, -0.1563033, 0.1569683
5: -0.0991381, 0.1209499, -0.0993342, 0.1228417, -0.2206618, 0.2189431
6: -0.1087681, 0.0952346, -0.1089060, 0.0963833, -0.2036864, 0.2033080
7: -0.0862061, 0.0896484, -0.0870833, 0.0898417, -0.1760478, 0.1767317
8: 0.5175176, 1.1331558, 0.5133498, 1.1333268, -0.5883665, 0.5959589
9: -0.0773586, 0.1615027, -0.0775784, 0.1626870, -0.2259800, 0.2240859

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3598784, upper bound: 0.3211620
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3598784, upper bound: 0.3740012
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1886644, 0.2059052, -0.1721372, 0.2339249, -0.4191089, 0.3745607
1: -0.0855726, 0.0766896, -0.0787497, 0.0829208, -0.1684934, 0.1554393
2: -0.1314478, 0.1169224, -0.1161624, 0.1203677, -0.2518154, 0.2330847
3: -0.1061568, 0.1538208, -0.0926231, 0.1637407, -0.2698975, 0.2464439
4: -0.0829563, 0.0734682, -0.0901669, 0.0663936, -0.1490783, 0.1630499
5: -0.0991381, 0.1209499, -0.0876733, 0.1310285, -0.2291198, 0.2074653
6: -0.1087681, 0.0952346, -0.0979364, 0.1014621, -0.2069250, 0.1924136
7: -0.0862061, 0.0896484, -0.0910093, 0.0794143, -0.1656204, 0.1806577
8: 0.5175176, 1.1331558, 0.4839187, 1.1091716, -0.5675201, 0.6311097
9: -0.0773586, 0.1615027, -0.0644370, 0.1676409, -0.2310473, 0.2118228

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3572984, upper bound: 0.3314487
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3087946, upper bound: 0.3189987
time: 0.64 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.57 seconds
IS_A1_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.57
Output dim: 8, lower bound: -0.3429854, upper bound: 0.3129795
IS_A1_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.57
Output dim: 8, lower bound: -0.3429854, upper bound: 0.3129795
IS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.57
Output dim: 8, lower bound: -0.3094543, upper bound: 0.3202145
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 8, lower bound: -0.3094543, upper bound: 0.3563497
IS_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.57
Output dim: 8, lower bound: -0.3426430, upper bound: 0.3247033
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 8, lower bound: -0.3426430, upper bound: 0.3838805
IS_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.57
Output dim: 8, lower bound: -0.2994156, upper bound: 0.2697806
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 8, lower bound: -0.3479309, upper bound: 0.3609453
IS_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.57
Output dim: 8, lower bound: -0.2934464, upper bound: 0.2685264
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 8, lower bound: -0.3451735, upper bound: 0.3609453
IS_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.57
Output dim: 8, lower bound: -0.3441772, upper bound: 0.3486989
IS_A1_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.57
Output dim: 8, lower bound: -0.3441772, upper bound: 0.3459820
IS_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.57
Output dim: 8, lower bound: -0.3290531, upper bound: 0.3281036
IS_A1_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.57
Output dim: 8, lower bound: -0.2725724, upper bound: 0.3113114
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3584130
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3618980
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3584130
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3618980
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 8, lower bound: -0.3563497, upper bound: 0.3127464
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 8, lower bound: -0.3563497, upper bound: 0.3710521
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 8, lower bound: -0.3609453, upper bound: 0.3479309
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 8, lower bound: -0.3609453, upper bound: 0.3451735
IS_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.57
Output dim: 8, lower bound: -0.3146726, upper bound: 0.2602647
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 8, lower bound: -0.3628378, upper bound: 0.3439668
IS_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.57
Output dim: 8, lower bound: -0.3151583, upper bound: 0.2600348
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 8, lower bound: -0.3653816, upper bound: 0.3439668
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 8, lower bound: -0.3584130, upper bound: 0.3427173
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 8, lower bound: -0.3584130, upper bound: 0.3427173
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 8, lower bound: -0.3618980, upper bound: 0.3427173
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 8, lower bound: -0.3618980, upper bound: 0.3427173
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 8, lower bound: -0.3598784, upper bound: 0.3211620
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 8, lower bound: -0.3598784, upper bound: 0.3740012
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.57
Output dim: 8, lower bound: -0.3572984, upper bound: 0.3314487
IS_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.57
Output dim: 8, lower bound: -0.3087946, upper bound: 0.3189987

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1899755, 0.1435920, -0.1886644, 0.2059052, -0.3925754, 0.3288960
1: -0.0847272, 0.0756130, -0.0855726, 0.0766896, -0.1614169, 0.1611856
2: -0.1342172, 0.1120768, -0.1314478, 0.1169224, -0.2511396, 0.2435246
3: -0.1098307, 0.1198920, -0.1061568, 0.1538208, -0.2636515, 0.2260488
4: -0.0672577, 0.0727668, -0.0829563, 0.0734682, -0.1407259, 0.1554698
5: -0.1002874, 0.1072509, -0.0991381, 0.1209499, -0.2206233, 0.2063890
6: -0.1103239, 0.0891246, -0.1087681, 0.0952346, -0.2047760, 0.1978927
7: -0.0697513, 0.0895727, -0.0862061, 0.0896484, -0.1593997, 0.1757788
8: 0.6093661, 1.1399536, 0.5175176, 1.1331558, -0.4934378, 0.5983143
9: -0.0776297, 0.1388457, -0.0773586, 0.1615027, -0.2253104, 0.2110059

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 142
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 177
type: A, layer: 3, pos: 59
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 248

Time for candidate selection: 2.16 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2937175, upper bound: 0.3425415
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2952297, upper bound: 0.3432251
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.1775892, 0.1604986, -0.1886644, 0.2059052, -0.3797312, 0.3442516
1: -0.0783708, 0.0696102, -0.0855726, 0.0766896, -0.1550605, 0.1551827
2: -0.1232215, 0.1047996, -0.1314478, 0.1169224, -0.2401439, 0.2362474
3: -0.1010032, 0.1423073, -0.1061568, 0.1538208, -0.2548241, 0.2484641
4: -0.0692312, 0.0664058, -0.0829563, 0.0734682, -0.1421834, 0.1490516
5: -0.0916116, 0.1089798, -0.0991381, 0.1209499, -0.2111367, 0.2074292
6: -0.1023371, 0.0872673, -0.1087681, 0.0952346, -0.1967221, 0.1959341
7: -0.0803046, 0.0805327, -0.0862061, 0.0896484, -0.1699530, 0.1667388
8: 0.5658924, 1.1247110, 0.5175176, 1.1331558, -0.5323586, 0.5790386
9: -0.0669297, 0.1527999, -0.0773586, 0.1615027, -0.2136373, 0.2146662

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 59
type: A, layer: 3, pos: 255
type: A, layer: 3, pos: 142
type: A, layer: 3, pos: 177
type: A, layer: 3, pos: 248
type: A, layer: 3, pos: 42

Time for candidate selection: 2.17 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3283572, upper bound: 0.3666448
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3274282, upper bound: 0.3669637
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1574272, 0.1647350, -0.1886644, 0.2059052, -0.3596785, 0.3489084
1: -0.0686271, 0.0689221, -0.0855726, 0.0766896, -0.1453167, 0.1544946
2: -0.1053974, 0.0966751, -0.1314478, 0.1169224, -0.2223198, 0.2281229
3: -0.0864605, 0.1418168, -0.1061568, 0.1538208, -0.2402813, 0.2479736
4: -0.0690144, 0.0566287, -0.0829563, 0.0734682, -0.1418262, 0.1392893
5: -0.0774258, 0.1084655, -0.0991381, 0.1209499, -0.1970461, 0.2047650
6: -0.0898071, 0.0869272, -0.1087681, 0.0952346, -0.1841819, 0.1923144
7: -0.0800737, 0.0672017, -0.0862061, 0.0896484, -0.1697221, 0.1534078
8: 0.5648291, 1.0987828, 0.5175176, 1.1331558, -0.5391500, 0.5539246
9: -0.0507631, 0.1523322, -0.0773586, 0.1615027, -0.1976759, 0.2150379

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2893766, upper bound: 0.3324947
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 59
type: A, layer: 3, pos: 142
type: A, layer: 3, pos: 255
type: A, layer: 3, pos: 177
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 121
type: A, layer: 3, pos: 248
type: A, layer: 3, pos: 180

Time for candidate selection: 4.82 seconds

### Candidate
type: A, layer: 3, pos: 230

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3405226, upper bound: 0.3412990
time: 1.08 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3292032, upper bound: 0.3445455
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.1600223, 0.1761911, -0.1886644, 0.2059052, -0.3625822, 0.3601599
1: -0.0703633, 0.0695929, -0.0855726, 0.0766896, -0.1470529, 0.1551654
2: -0.1073237, 0.0999089, -0.1314478, 0.1169224, -0.2242461, 0.2313567
3: -0.0875995, 0.1429007, -0.1061568, 0.1538208, -0.2414203, 0.2490575
4: -0.0714792, 0.0582858, -0.0829563, 0.0734682, -0.1442755, 0.1409877
5: -0.0791898, 0.1096610, -0.0991381, 0.1209499, -0.1990914, 0.2062328
6: -0.0912108, 0.0878678, -0.1087681, 0.0952346, -0.1857650, 0.1932054
7: -0.0807419, 0.0691984, -0.0862061, 0.0896484, -0.1703903, 0.1554046
8: 0.5550525, 1.1008596, 0.5175176, 1.1331558, -0.5494637, 0.5588233
9: -0.0531513, 0.1534001, -0.0773586, 0.1615027, -0.2009378, 0.2157223

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2831905, upper bound: 0.3265820
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 59
type: A, layer: 3, pos: 142
type: A, layer: 3, pos: 255
type: A, layer: 3, pos: 177
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 121
type: A, layer: 3, pos: 248
type: A, layer: 3, pos: 180

Time for candidate selection: 4.94 seconds

### Candidate
type: A, layer: 3, pos: 230

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3376492, upper bound: 0.3412990
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3255335, upper bound: 0.3445101
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1574272, 0.1647350, -0.1660702, 0.2018342, -0.3551338, 0.3262532
1: -0.0686271, 0.0689221, -0.0747557, 0.0741434, -0.1427705, 0.1436778
2: -0.1053974, 0.0966751, -0.1117253, 0.1088224, -0.2142198, 0.2084005
3: -0.0864605, 0.1418168, -0.0900854, 0.1500339, -0.2364944, 0.2319021
4: -0.0690144, 0.0566287, -0.0792971, 0.0625792, -0.1309292, 0.1353558
5: -0.0774258, 0.1084655, -0.0834441, 0.1170112, -0.1925342, 0.1888929
6: -0.0898071, 0.0869272, -0.0948899, 0.0926307, -0.1790389, 0.1783733
7: -0.0800737, 0.0672017, -0.0843168, 0.0746323, -0.1547060, 0.1515185
8: 0.5648291, 1.0987828, 0.5274178, 1.1043098, -0.5106983, 0.5472000
9: -0.0507631, 0.1523322, -0.0594005, 0.1584760, -0.1946074, 0.1967568

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2141893, upper bound: 0.2962512
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 59
type: A, layer: 3, pos: 142
type: A, layer: 3, pos: 255
type: A, layer: 3, pos: 177
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 121
type: A, layer: 3, pos: 248
type: A, layer: 3, pos: 180

Time for candidate selection: 4.86 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3445386, upper bound: 0.3595635
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3584130
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1574272, 0.1647350, -0.1708696, 0.2229829, -0.3771502, 0.3312626
1: -0.0686271, 0.0689221, -0.0778513, 0.0773110, -0.1459381, 0.1467734
2: -0.1053974, 0.0966751, -0.1154248, 0.1155212, -0.2209186, 0.2121000
3: -0.0864605, 0.1418168, -0.0923715, 0.1550202, -0.2414807, 0.2341883
4: -0.0690144, 0.0566287, -0.0852941, 0.0655414, -0.1339202, 0.1414983
5: -0.0774258, 0.1084655, -0.0867782, 0.1221924, -0.1985891, 0.1924735
6: -0.0898071, 0.0869272, -0.0974704, 0.0960724, -0.1826857, 0.1810776
7: -0.0800737, 0.0672017, -0.0868933, 0.0782940, -0.1583677, 0.1540950
8: 0.5648291, 1.0987828, 0.5048499, 1.1085424, -0.5167670, 0.5753944
9: -0.0507631, 0.1523322, -0.0635459, 0.1620800, -0.1992641, 0.2015342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2141893, upper bound: 0.2962512
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 59
type: A, layer: 3, pos: 142
type: A, layer: 3, pos: 255
type: A, layer: 3, pos: 177
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 121
type: A, layer: 3, pos: 248
type: A, layer: 3, pos: 180

Time for candidate selection: 4.80 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3445386, upper bound: 0.3618980
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3618980
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.1600223, 0.1761911, -0.1660702, 0.2018342, -0.3579617, 0.3380330
1: -0.0703633, 0.0695929, -0.0747557, 0.0741434, -0.1445067, 0.1443486
2: -0.1073237, 0.0999089, -0.1117253, 0.1088224, -0.2161461, 0.2116342
3: -0.0875995, 0.1429007, -0.0900854, 0.1500339, -0.2376333, 0.2329861
4: -0.0714792, 0.0582858, -0.0792971, 0.0625792, -0.1334496, 0.1370451
5: -0.0791898, 0.1096610, -0.0834441, 0.1170112, -0.1944954, 0.1907833
6: -0.0912108, 0.0878678, -0.0948899, 0.0926307, -0.1805897, 0.1795070
7: -0.0807419, 0.0691984, -0.0843168, 0.0746323, -0.1553742, 0.1535152
8: 0.5550525, 1.1008596, 0.5274178, 1.1043098, -0.5242290, 0.5512948
9: -0.0531513, 0.1534001, -0.0594005, 0.1584760, -0.1975967, 0.1986885

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 59
type: A, layer: 3, pos: 142
type: A, layer: 3, pos: 255
type: A, layer: 3, pos: 177
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 121
type: A, layer: 3, pos: 248
type: A, layer: 3, pos: 180

Time for candidate selection: 4.34 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3584130
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3584130
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.1600223, 0.1761911, -0.1708696, 0.2229829, -0.3792681, 0.3424203
1: -0.0703633, 0.0695929, -0.0778513, 0.0773110, -0.1476743, 0.1474442
2: -0.1073237, 0.0999089, -0.1154248, 0.1155212, -0.2228449, 0.2153337
3: -0.0875995, 0.1429007, -0.0923715, 0.1550202, -0.2426196, 0.2352722
4: -0.0714792, 0.0582858, -0.0852941, 0.0655414, -0.1363571, 0.1430888
5: -0.0791898, 0.1096610, -0.0867782, 0.1221924, -0.2000332, 0.1938493
6: -0.0912108, 0.0878678, -0.0974704, 0.0960724, -0.1839260, 0.1819101
7: -0.0807419, 0.0691984, -0.0868933, 0.0782940, -0.1590359, 0.1560917
8: 0.5550525, 1.1008596, 0.5048499, 1.1085424, -0.5262451, 0.5747843
9: -0.0531513, 0.1534001, -0.0635459, 0.1620800, -0.2005690, 0.2019295

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 59
type: A, layer: 3, pos: 142
type: A, layer: 3, pos: 255
type: A, layer: 3, pos: 177
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 121
type: A, layer: 3, pos: 248
type: A, layer: 3, pos: 180

Time for candidate selection: 4.21 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3618980
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3618980
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.1886644, 0.2059052, -0.1899755, 0.1435920, -0.3288960, 0.3925755
1: -0.0855726, 0.0766896, -0.0847272, 0.0756130, -0.1611856, 0.1614169
2: -0.1314478, 0.1169224, -0.1342172, 0.1120768, -0.2435246, 0.2511396
3: -0.1061568, 0.1538208, -0.1098307, 0.1198920, -0.2260488, 0.2636515
4: -0.0829563, 0.0734682, -0.0672577, 0.0727668, -0.1554698, 0.1407259
5: -0.0991381, 0.1209499, -0.1002874, 0.1072509, -0.2063890, 0.2206233
6: -0.1087681, 0.0952346, -0.1103239, 0.0891246, -0.1978927, 0.2047759
7: -0.0862061, 0.0896484, -0.0697513, 0.0895727, -0.1757788, 0.1593997
8: 0.5175176, 1.1331558, 0.6093661, 1.1399536, -0.5983140, 0.4934380
9: -0.0773586, 0.1615027, -0.0776297, 0.1388457, -0.2110059, 0.2253105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=23, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 59
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 255
type: A, layer: 3, pos: 142
type: A, layer: 3, pos: 177
type: A, layer: 3, pos: 140
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 248

Time for candidate selection: 2.19 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3399636, upper bound: 0.2984377
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3432251, upper bound: 0.2985930
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.1886644, 0.2059052, -0.1775892, 0.1604986, -0.3442516, 0.3797313
1: -0.0855726, 0.0766896, -0.0783708, 0.0696102, -0.1551827, 0.1550605
2: -0.1314478, 0.1169224, -0.1232215, 0.1047996, -0.2362474, 0.2401439
3: -0.1061568, 0.1538208, -0.1010032, 0.1423073, -0.2484641, 0.2548241
4: -0.0829563, 0.0734682, -0.0692312, 0.0664058, -0.1490516, 0.1421834
5: -0.0991381, 0.1209499, -0.0916116, 0.1089798, -0.2074291, 0.2111366
6: -0.1087681, 0.0952346, -0.1023371, 0.0872673, -0.1959341, 0.1967221
7: -0.0862061, 0.0896484, -0.0803046, 0.0805327, -0.1667388, 0.1699530
8: 0.5175176, 1.1331558, 0.5658924, 1.1247110, -0.5790384, 0.5323586
9: -0.0773586, 0.1615027, -0.0669297, 0.1527999, -0.2146662, 0.2136373

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 59
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 255
type: A, layer: 3, pos: 142
type: A, layer: 3, pos: 177
type: A, layer: 3, pos: 140
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 248

Time for candidate selection: 2.17 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3399636, upper bound: 0.3534707
time: 0.95 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3432251, upper bound: 0.3535647
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.1886644, 0.2059052, -0.1574272, 0.1647350, -0.3489085, 0.3596785
1: -0.0855726, 0.0766896, -0.0686271, 0.0689221, -0.1544946, 0.1453167
2: -0.1314478, 0.1169224, -0.1053974, 0.0966751, -0.2281229, 0.2223198
3: -0.1061568, 0.1538208, -0.0864605, 0.1418168, -0.2479736, 0.2402813
4: -0.0829563, 0.0734682, -0.0690144, 0.0566287, -0.1392893, 0.1418262
5: -0.0991381, 0.1209499, -0.0774258, 0.1084655, -0.2047649, 0.1970460
6: -0.1087681, 0.0952346, -0.0898071, 0.0869272, -0.1923144, 0.1841819
7: -0.0862061, 0.0896484, -0.0800737, 0.0672017, -0.1534078, 0.1697221
8: 0.5175176, 1.1331558, 0.5648291, 1.0987828, -0.5539246, 0.5391498
9: -0.0773586, 0.1615027, -0.0507631, 0.1523322, -0.2150380, 0.1976759

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 59
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 255
type: A, layer: 3, pos: 142
type: A, layer: 3, pos: 177
type: A, layer: 3, pos: 140
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 248

Time for candidate selection: 2.24 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3571132, upper bound: 0.3451735
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3571132, upper bound: 0.3451735
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.1886644, 0.2059052, -0.1600223, 0.1761911, -0.3601598, 0.3625822
1: -0.0855726, 0.0766896, -0.0703633, 0.0695929, -0.1551654, 0.1470529
2: -0.1314478, 0.1169224, -0.1073237, 0.0999089, -0.2313567, 0.2242461
3: -0.1061568, 0.1538208, -0.0875995, 0.1429007, -0.2490575, 0.2414203
4: -0.0829563, 0.0734682, -0.0714792, 0.0582858, -0.1409878, 0.1442755
5: -0.0991381, 0.1209499, -0.0791898, 0.1096610, -0.2062328, 0.1990914
6: -0.1087681, 0.0952346, -0.0912108, 0.0878678, -0.1932054, 0.1857650
7: -0.0862061, 0.0896484, -0.0807419, 0.0691984, -0.1554046, 0.1703903
8: 0.5175176, 1.1331558, 0.5550525, 1.1008596, -0.5588231, 0.5494637
9: -0.0773586, 0.1615027, -0.0531513, 0.1534001, -0.2157223, 0.2009377

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 59
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 255
type: A, layer: 3, pos: 142
type: A, layer: 3, pos: 177
type: A, layer: 3, pos: 140
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 248

Time for candidate selection: 2.18 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3571132, upper bound: 0.3451735
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3571132, upper bound: 0.3451735
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1660702, 0.2018342, -0.1775892, 0.1604986, -0.3217039, 0.3753216
1: -0.0747557, 0.0741434, -0.0783708, 0.0696102, -0.1443659, 0.1525143
2: -0.1117253, 0.1088224, -0.1232215, 0.1047996, -0.2165250, 0.2320440
3: -0.0900854, 0.1500339, -0.1010032, 0.1423073, -0.2323926, 0.2510371
4: -0.0792971, 0.0625792, -0.0692312, 0.0664058, -0.1451367, 0.1313004
5: -0.0834441, 0.1170112, -0.0916116, 0.1089798, -0.1916725, 0.2067394
6: -0.0948899, 0.0926307, -0.1023371, 0.0872673, -0.1819856, 0.1916257
7: -0.0843168, 0.0746323, -0.0803046, 0.0805327, -0.1648495, 0.1549369
8: 0.5274178, 1.1043098, 0.5658924, 1.1247110, -0.5728986, 0.5045042
9: -0.0594005, 0.1584760, -0.0669297, 0.1527999, -0.1967320, 0.2110068

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2924999, upper bound: 0.3038757
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 59
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 142
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 255
type: A, layer: 3, pos: 140
type: A, layer: 3, pos: 177
type: A, layer: 3, pos: 121
type: A, layer: 3, pos: 248
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 91
type: A, layer: 3, pos: 180

Time for candidate selection: 4.99 seconds

### Candidate
type: A, layer: 3, pos: 230

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3550109, upper bound: 0.3219343
time: 0.92 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3472517, upper bound: 0.3259502
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.1708696, 0.2229829, -0.1775892, 0.1604986, -0.3267319, 0.3967973
1: -0.0778513, 0.0773110, -0.0783708, 0.0696102, -0.1474615, 0.1556818
2: -0.1154248, 0.1155212, -0.1232215, 0.1047996, -0.2202245, 0.2387427
3: -0.0923715, 0.1550202, -0.1010032, 0.1423073, -0.2346788, 0.2560234
4: -0.0852941, 0.0655414, -0.0692312, 0.0664058, -0.1512043, 0.1342945
5: -0.0867782, 0.1221924, -0.0916116, 0.1089798, -0.1952778, 0.2124107
6: -0.0974704, 0.0960724, -0.1023371, 0.0872673, -0.1847234, 0.1950311
7: -0.0868933, 0.0782940, -0.0803046, 0.0805327, -0.1674260, 0.1585986
8: 0.5048499, 1.1085424, 0.5658924, 1.1247110, -0.5976069, 0.5110879
9: -0.0635459, 0.1620800, -0.0669297, 0.1527999, -0.2016904, 0.2143171

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2924379, upper bound: 0.3034252
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 59
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 142
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 255
type: A, layer: 3, pos: 140
type: A, layer: 3, pos: 177
type: A, layer: 3, pos: 248
type: A, layer: 3, pos: 121
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 91
type: A, layer: 3, pos: 180

Time for candidate selection: 5.02 seconds

### Candidate
type: A, layer: 3, pos: 230

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3575774, upper bound: 0.3219343
time: 0.98 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3496493, upper bound: 0.3258807
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1660702, 0.2018342, -0.1574272, 0.1647350, -0.3262532, 0.3551338
1: -0.0747557, 0.0741434, -0.0686271, 0.0689221, -0.1436778, 0.1427705
2: -0.1117253, 0.1088224, -0.1053974, 0.0966751, -0.2084005, 0.2142198
3: -0.0900854, 0.1500339, -0.0864605, 0.1418168, -0.2319021, 0.2364944
4: -0.0792971, 0.0625792, -0.0690144, 0.0566287, -0.1353558, 0.1309292
5: -0.0834441, 0.1170112, -0.0774258, 0.1084655, -0.1888929, 0.1925342
6: -0.0948899, 0.0926307, -0.0898071, 0.0869272, -0.1783734, 0.1790389
7: -0.0843168, 0.0746323, -0.0800737, 0.0672017, -0.1515185, 0.1547060
8: 0.5274178, 1.1043098, 0.5648291, 1.0987828, -0.5471997, 0.5106981
9: -0.0594005, 0.1584760, -0.0507631, 0.1523322, -0.1967569, 0.1946074

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 59
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 142
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 255
type: A, layer: 3, pos: 140
type: A, layer: 3, pos: 177
type: A, layer: 3, pos: 121
type: A, layer: 3, pos: 248
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 91
type: A, layer: 3, pos: 180

Time for candidate selection: 4.20 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3584130, upper bound: 0.3427173
time: 0.98 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3584130, upper bound: 0.3427173
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1660702, 0.2018342, -0.1600223, 0.1761911, -0.3380329, 0.3579617
1: -0.0747557, 0.0741434, -0.0703633, 0.0695929, -0.1443486, 0.1445067
2: -0.1117253, 0.1088224, -0.1073237, 0.0999089, -0.2116342, 0.2161461
3: -0.0900854, 0.1500339, -0.0875995, 0.1429007, -0.2329861, 0.2376333
4: -0.0792971, 0.0625792, -0.0714792, 0.0582858, -0.1370451, 0.1334496
5: -0.0834441, 0.1170112, -0.0791898, 0.1096610, -0.1907832, 0.1944954
6: -0.0948899, 0.0926307, -0.0912108, 0.0878678, -0.1795070, 0.1805897
7: -0.0843168, 0.0746323, -0.0807419, 0.0691984, -0.1535152, 0.1553742
8: 0.5274178, 1.1043098, 0.5550525, 1.1008596, -0.5512948, 0.5242290
9: -0.0594005, 0.1584760, -0.0531513, 0.1534001, -0.1986885, 0.1975968

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 59
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 142
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 255
type: A, layer: 3, pos: 140
type: A, layer: 3, pos: 177
type: A, layer: 3, pos: 121
type: A, layer: 3, pos: 248
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 91
type: A, layer: 3, pos: 180

Time for candidate selection: 4.19 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3584130, upper bound: 0.3427173
time: 1.00 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3584130, upper bound: 0.3427173
time: 0.95 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.1708696, 0.2229829, -0.1574272, 0.1647350, -0.3312626, 0.3771502
1: -0.0778513, 0.0773110, -0.0686271, 0.0689221, -0.1467734, 0.1459381
2: -0.1154248, 0.1155212, -0.1053974, 0.0966751, -0.2121000, 0.2209186
3: -0.0923715, 0.1550202, -0.0864605, 0.1418168, -0.2341883, 0.2414807
4: -0.0852941, 0.0655414, -0.0690144, 0.0566287, -0.1414983, 0.1339202
5: -0.0867782, 0.1221924, -0.0774258, 0.1084655, -0.1924734, 0.1985891
6: -0.0974704, 0.0960724, -0.0898071, 0.0869272, -0.1810776, 0.1826857
7: -0.0868933, 0.0782940, -0.0800737, 0.0672017, -0.1540950, 0.1583677
8: 0.5048499, 1.1085424, 0.5648291, 1.0987828, -0.5753942, 0.5167670
9: -0.0635459, 0.1620800, -0.0507631, 0.1523322, -0.2015341, 0.1992642

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 59
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 142
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 255
type: A, layer: 3, pos: 140
type: A, layer: 3, pos: 177
type: A, layer: 3, pos: 248
type: A, layer: 3, pos: 121
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 91
type: A, layer: 3, pos: 180

Time for candidate selection: 4.39 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3584130, upper bound: 0.3427173
time: 0.92 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3618979, upper bound: 0.3427173
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.1708696, 0.2229829, -0.1600223, 0.1761911, -0.3424202, 0.3792679
1: -0.0778513, 0.0773110, -0.0703633, 0.0695929, -0.1474442, 0.1476743
2: -0.1154248, 0.1155212, -0.1073237, 0.0999089, -0.2153337, 0.2228449
3: -0.0923715, 0.1550202, -0.0875995, 0.1429007, -0.2352722, 0.2426196
4: -0.0852941, 0.0655414, -0.0714792, 0.0582858, -0.1430888, 0.1363571
5: -0.0867782, 0.1221924, -0.0791898, 0.1096610, -0.1938493, 0.2000332
6: -0.0974704, 0.0960724, -0.0912108, 0.0878678, -0.1819101, 0.1839260
7: -0.0868933, 0.0782940, -0.0807419, 0.0691984, -0.1560917, 0.1590359
8: 0.5048499, 1.1085424, 0.5550525, 1.1008596, -0.5747843, 0.5262451
9: -0.0635459, 0.1620800, -0.0531513, 0.1534001, -0.2019295, 0.2005691

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 59
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 142
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 255
type: A, layer: 3, pos: 140
type: A, layer: 3, pos: 177
type: A, layer: 3, pos: 248
type: A, layer: 3, pos: 121
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 91
type: A, layer: 3, pos: 180

Time for candidate selection: 4.25 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3584130, upper bound: 0.3427173
time: 0.95 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3618979, upper bound: 0.3427173
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.1886644, 0.2059052, -0.2004538, 0.1907003, -0.3765460, 0.4030333
1: -0.0855726, 0.0766896, -0.0916685, 0.0822144, -0.1677869, 0.1683581
2: -0.1314478, 0.1169224, -0.1421823, 0.1213505, -0.2527983, 0.2591047
3: -0.1061568, 0.1538208, -0.1149285, 0.1312140, -0.2373708, 0.2687493
4: -0.0829563, 0.0734682, -0.0802339, 0.0795387, -0.1622391, 0.1537021
5: -0.0991381, 0.1209499, -0.1075740, 0.1133828, -0.2125209, 0.2282743
6: -0.1087681, 0.0952346, -0.1167236, 0.0957305, -0.2044986, 0.2111741
7: -0.0862061, 0.0896484, -0.0756733, 0.0986896, -0.1848957, 0.1653217
8: 0.5175176, 1.1331558, 0.5630162, 1.1479977, -0.6073196, 0.5456460
9: -0.0773586, 0.1615027, -0.0880105, 0.1476709, -0.2192700, 0.2355457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 59
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 255
type: A, layer: 3, pos: 142
type: A, layer: 3, pos: 177
type: A, layer: 3, pos: 140
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 248

Time for candidate selection: 2.20 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3427390, upper bound: 0.3068470
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3476431, upper bound: 0.3082602
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.1886644, 0.2059052, -0.1886644, 0.2059052, -0.3908117, 0.3908119
1: -0.0855726, 0.0766896, -0.0855726, 0.0766896, -0.1622622, 0.1622622
2: -0.1314478, 0.1169224, -0.1314478, 0.1169224, -0.2483702, 0.2483702
3: -0.1061568, 0.1538208, -0.1061568, 0.1538208, -0.2599776, 0.2599776
4: -0.0829563, 0.0734682, -0.0829563, 0.0734682, -0.1561147, 0.1561147
5: -0.0991381, 0.1209499, -0.0991381, 0.1209499, -0.2187878, 0.2187878
6: -0.1087681, 0.0952346, -0.1087681, 0.0952346, -0.2031702, 0.2031701
7: -0.0862061, 0.0896484, -0.0862061, 0.0896484, -0.1758545, 0.1758545
8: 0.5175176, 1.1331558, 0.5175176, 1.1331558, -0.5881937, 0.5881934
9: -0.0773586, 0.1615027, -0.0773586, 0.1615027, -0.2238629, 0.2238629

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 59
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 255
type: A, layer: 3, pos: 142
type: A, layer: 3, pos: 177
type: A, layer: 3, pos: 140
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 248

Time for candidate selection: 2.17 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3427390, upper bound: 0.3564572
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3476431, upper bound: 0.3582547
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.1886644, 0.2059052, -0.1683835, 0.2251082, -0.4098923, 0.3708140
1: -0.0855726, 0.0766896, -0.0762336, 0.0821216, -0.1676942, 0.1529232
2: -0.1314478, 0.1169224, -0.1131642, 0.1173433, -0.2487911, 0.2300866
3: -0.1061568, 0.1538208, -0.0907147, 0.1624790, -0.2686358, 0.2445356
4: -0.0829563, 0.0734682, -0.0877326, 0.0638773, -0.1465629, 0.1605596
5: -0.0991381, 0.1209499, -0.0849793, 0.1297007, -0.2275524, 0.2047612
6: -0.1087681, 0.0952346, -0.0955993, 0.1005463, -0.2059919, 0.1900974
7: -0.0862061, 0.0896484, -0.0903383, 0.0763131, -0.1625192, 0.1799868
8: 0.5175176, 1.1331558, 0.4909878, 1.1062018, -0.5640874, 0.6228530
9: -0.0773586, 0.1615027, -0.0607465, 0.1666286, -0.2299698, 0.2081959

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3087946, upper bound: 0.3189987
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3087946, upper bound: 0.3189987
time: 0.63 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 2.56 seconds
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 8, lower bound: -0.2937175, upper bound: 0.3425415
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 8, lower bound: -0.2952297, upper bound: 0.3432251
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 8, lower bound: -0.3283572, upper bound: 0.3666448
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 8, lower bound: -0.3274282, upper bound: 0.3669637
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 8, lower bound: -0.3405226, upper bound: 0.3412990
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 8, lower bound: -0.3292032, upper bound: 0.3445455
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 8, lower bound: -0.3376492, upper bound: 0.3412990
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 8, lower bound: -0.3255335, upper bound: 0.3445101
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 8, lower bound: -0.3445386, upper bound: 0.3595635
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3584130
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 8, lower bound: -0.3445386, upper bound: 0.3618980
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3618980
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3584130
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3584130
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3618980
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3618980
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 8, lower bound: -0.3399636, upper bound: 0.2984377
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 8, lower bound: -0.3432251, upper bound: 0.2985930
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 8, lower bound: -0.3399636, upper bound: 0.3534707
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 8, lower bound: -0.3432251, upper bound: 0.3535647
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 8, lower bound: -0.3571132, upper bound: 0.3451735
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 8, lower bound: -0.3571132, upper bound: 0.3451735
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 8, lower bound: -0.3571132, upper bound: 0.3451735
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 8, lower bound: -0.3571132, upper bound: 0.3451735
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 8, lower bound: -0.3550109, upper bound: 0.3219343
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 8, lower bound: -0.3472517, upper bound: 0.3259502
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 8, lower bound: -0.3575774, upper bound: 0.3219343
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 8, lower bound: -0.3496493, upper bound: 0.3258807
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 8, lower bound: -0.3584130, upper bound: 0.3427173
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 8, lower bound: -0.3584130, upper bound: 0.3427173
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 8, lower bound: -0.3584130, upper bound: 0.3427173
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 8, lower bound: -0.3584130, upper bound: 0.3427173
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 8, lower bound: -0.3584130, upper bound: 0.3427173
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 8, lower bound: -0.3618979, upper bound: 0.3427173
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 8, lower bound: -0.3584130, upper bound: 0.3427173
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 8, lower bound: -0.3618979, upper bound: 0.3427173
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 8, lower bound: -0.3427390, upper bound: 0.3068470
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 8, lower bound: -0.3476431, upper bound: 0.3082602
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 8, lower bound: -0.3427390, upper bound: 0.3564572
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 8, lower bound: -0.3476431, upper bound: 0.3582547
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 8, lower bound: -0.3087946, upper bound: 0.3189987
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.56
Output dim: 8, lower bound: -0.3087946, upper bound: 0.3189987

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1748508, 0.1390383, -0.1886528, 0.2058144, -0.3767657, 0.3222039
1: -0.0763320, 0.0676586, -0.0855639, 0.0766658, -0.1529978, 0.1532226
2: -0.1214549, 0.1019824, -0.1314403, 0.1168991, -0.2383540, 0.2334228
3: -0.1002031, 0.1324989, -0.1061534, 0.1537807, -0.2539838, 0.2377524
4: -0.0617115, 0.0643708, -0.0829258, 0.0734600, -0.1349241, 0.1469676
5: -0.0898271, 0.1020116, -0.0991305, 0.1209091, -0.2092223, 0.2011421
6: -0.1008040, 0.0808714, -0.1087622, 0.0952089, -0.1951211, 0.1896337
7: -0.0755110, 0.0780726, -0.0861866, 0.0896389, -0.1651499, 0.1642592
8: 0.5941271, 1.1236662, 0.5176398, 1.1331506, -0.5004909, 0.5757582
9: -0.0641801, 0.1459204, -0.0773478, 0.1614756, -0.2103270, 0.2079321

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3432554, upper bound: 0.3378745
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3212425, upper bound: 0.3362557
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.1761420, 0.1461693, -0.1886413, 0.2056740, -0.3780637, 0.3295193
1: -0.0773602, 0.0686495, -0.0855564, 0.0765909, -0.1539511, 0.1542059
2: -0.1223146, 0.1033744, -0.1314345, 0.1168620, -0.2391766, 0.2348088
3: -0.1005974, 0.1318670, -0.1061515, 0.1536440, -0.2542414, 0.2375227
4: -0.0632209, 0.0653967, -0.0828646, 0.0734528, -0.1366023, 0.1479558
5: -0.0906619, 0.1017884, -0.0991227, 0.1207705, -0.2101502, 0.2009111
6: -0.1016164, 0.0817272, -0.1087591, 0.0951246, -0.1959715, 0.1904862
7: -0.0753203, 0.0793914, -0.0861221, 0.0896333, -0.1649537, 0.1655135
8: 0.5904739, 1.1240447, 0.5179583, 1.1331449, -0.5050392, 0.5778315
9: -0.0656525, 0.1459402, -0.0773415, 0.1613887, -0.2122729, 0.2079087

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3383572, upper bound: 0.3352467
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3178940, upper bound: 0.3325054
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1574272, 0.1647350, -0.1660702, 0.2018342, -0.3551338, 0.3262532
1: -0.0686271, 0.0689221, -0.0747557, 0.0741434, -0.1427705, 0.1436778
2: -0.1053974, 0.0966751, -0.1117253, 0.1088224, -0.2142198, 0.2084005
3: -0.0864605, 0.1418168, -0.0900854, 0.1500339, -0.2364944, 0.2319021
4: -0.0690144, 0.0566287, -0.0792971, 0.0625792, -0.1309292, 0.1353558
5: -0.0774258, 0.1084655, -0.0834441, 0.1170112, -0.1925342, 0.1888929
6: -0.0898071, 0.0869272, -0.0948899, 0.0926307, -0.1790389, 0.1783733
7: -0.0800737, 0.0672017, -0.0843168, 0.0746323, -0.1547060, 0.1515185
8: 0.5648291, 1.0987828, 0.5274178, 1.1043098, -0.5106983, 0.5472000
9: -0.0507631, 0.1523322, -0.0594005, 0.1584760, -0.1946074, 0.1967568

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 59
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 142
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 255
type: B, layer: 3, pos: 140
type: B, layer: 3, pos: 177
type: B, layer: 3, pos: 121
type: B, layer: 3, pos: 248
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 91
type: B, layer: 3, pos: 180

Time for candidate selection: 4.15 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3584130
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3584130
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1600223, 0.1761911, -0.1660702, 0.2018342, -0.3579617, 0.3380330
1: -0.0703633, 0.0695929, -0.0747557, 0.0741434, -0.1445067, 0.1443486
2: -0.1073237, 0.0999089, -0.1117253, 0.1088224, -0.2161461, 0.2116342
3: -0.0875995, 0.1429007, -0.0900854, 0.1500339, -0.2376333, 0.2329861
4: -0.0714792, 0.0582858, -0.0792971, 0.0625792, -0.1334496, 0.1370451
5: -0.0791898, 0.1096610, -0.0834441, 0.1170112, -0.1944954, 0.1907833
6: -0.0912108, 0.0878678, -0.0948899, 0.0926307, -0.1805897, 0.1795070
7: -0.0807419, 0.0691984, -0.0843168, 0.0746323, -0.1553742, 0.1535152
8: 0.5550525, 1.1008596, 0.5274178, 1.1043098, -0.5242290, 0.5512948
9: -0.0531513, 0.1534001, -0.0594005, 0.1584760, -0.1975967, 0.1986885

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 59
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 142
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 255
type: B, layer: 3, pos: 140
type: B, layer: 3, pos: 177
type: B, layer: 3, pos: 121
type: B, layer: 3, pos: 248
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 91
type: B, layer: 3, pos: 180

Time for candidate selection: 4.22 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3584130
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3584130
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1574272, 0.1647350, -0.1708696, 0.2229829, -0.3771502, 0.3312626
1: -0.0686271, 0.0689221, -0.0778513, 0.0773110, -0.1459381, 0.1467734
2: -0.1053974, 0.0966751, -0.1154248, 0.1155212, -0.2209186, 0.2121000
3: -0.0864605, 0.1418168, -0.0923715, 0.1550202, -0.2414807, 0.2341883
4: -0.0690144, 0.0566287, -0.0852941, 0.0655414, -0.1339202, 0.1414983
5: -0.0774258, 0.1084655, -0.0867782, 0.1221924, -0.1985891, 0.1924735
6: -0.0898071, 0.0869272, -0.0974704, 0.0960724, -0.1826857, 0.1810776
7: -0.0800737, 0.0672017, -0.0868933, 0.0782940, -0.1583677, 0.1540950
8: 0.5648291, 1.0987828, 0.5048499, 1.1085424, -0.5167670, 0.5753944
9: -0.0507631, 0.1523322, -0.0635459, 0.1620800, -0.1992641, 0.2015342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 59
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 142
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 255
type: B, layer: 3, pos: 140
type: B, layer: 3, pos: 177
type: B, layer: 3, pos: 248
type: B, layer: 3, pos: 121
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 91
type: B, layer: 3, pos: 180

Time for candidate selection: 4.24 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3584130
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3618979
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1600223, 0.1761911, -0.1708696, 0.2229829, -0.3792681, 0.3424203
1: -0.0703633, 0.0695929, -0.0778513, 0.0773110, -0.1476743, 0.1474442
2: -0.1073237, 0.0999089, -0.1154248, 0.1155212, -0.2228449, 0.2153337
3: -0.0875995, 0.1429007, -0.0923715, 0.1550202, -0.2426196, 0.2352722
4: -0.0714792, 0.0582858, -0.0852941, 0.0655414, -0.1363571, 0.1430888
5: -0.0791898, 0.1096610, -0.0867782, 0.1221924, -0.2000332, 0.1938493
6: -0.0912108, 0.0878678, -0.0974704, 0.0960724, -0.1839260, 0.1819101
7: -0.0807419, 0.0691984, -0.0868933, 0.0782940, -0.1590359, 0.1560917
8: 0.5550525, 1.1008596, 0.5048499, 1.1085424, -0.5262451, 0.5747843
9: -0.0531513, 0.1534001, -0.0635459, 0.1620800, -0.2005690, 0.2019295

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 59
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 142
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 255
type: B, layer: 3, pos: 140
type: B, layer: 3, pos: 177
type: B, layer: 3, pos: 248
type: B, layer: 3, pos: 121
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 91
type: B, layer: 3, pos: 180

Time for candidate selection: 4.25 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3584130
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3618979
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1574272, 0.1647350, -0.1660702, 0.2018342, -0.3551338, 0.3262532
1: -0.0686271, 0.0689221, -0.0747557, 0.0741434, -0.1427705, 0.1436778
2: -0.1053974, 0.0966751, -0.1117253, 0.1088224, -0.2142198, 0.2084005
3: -0.0864605, 0.1418168, -0.0900854, 0.1500339, -0.2364944, 0.2319021
4: -0.0690144, 0.0566287, -0.0792971, 0.0625792, -0.1309292, 0.1353558
5: -0.0774258, 0.1084655, -0.0834441, 0.1170112, -0.1925342, 0.1888929
6: -0.0898071, 0.0869272, -0.0948899, 0.0926307, -0.1790389, 0.1783733
7: -0.0800737, 0.0672017, -0.0843168, 0.0746323, -0.1547060, 0.1515185
8: 0.5648291, 1.0987828, 0.5274178, 1.1043098, -0.5106983, 0.5472000
9: -0.0507631, 0.1523322, -0.0594005, 0.1584760, -0.1946074, 0.1967568

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 59
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 142
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 255
type: B, layer: 3, pos: 140
type: B, layer: 3, pos: 177
type: B, layer: 3, pos: 121
type: B, layer: 3, pos: 248
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 91
type: B, layer: 3, pos: 180

Time for candidate selection: 4.24 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3584130
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3584130
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.1600223, 0.1761911, -0.1660702, 0.2018342, -0.3579617, 0.3380330
1: -0.0703633, 0.0695929, -0.0747557, 0.0741434, -0.1445067, 0.1443486
2: -0.1073237, 0.0999089, -0.1117253, 0.1088224, -0.2161461, 0.2116342
3: -0.0875995, 0.1429007, -0.0900854, 0.1500339, -0.2376333, 0.2329861
4: -0.0714792, 0.0582858, -0.0792971, 0.0625792, -0.1334496, 0.1370451
5: -0.0791898, 0.1096610, -0.0834441, 0.1170112, -0.1944954, 0.1907833
6: -0.0912108, 0.0878678, -0.0948899, 0.0926307, -0.1805897, 0.1795070
7: -0.0807419, 0.0691984, -0.0843168, 0.0746323, -0.1553742, 0.1535152
8: 0.5550525, 1.1008596, 0.5274178, 1.1043098, -0.5242290, 0.5512948
9: -0.0531513, 0.1534001, -0.0594005, 0.1584760, -0.1975967, 0.1986885

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 59
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 142
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 255
type: B, layer: 3, pos: 140
type: B, layer: 3, pos: 177
type: B, layer: 3, pos: 121
type: B, layer: 3, pos: 248
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 91
type: B, layer: 3, pos: 180

Time for candidate selection: 4.22 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3584130
time: 1.09 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3584130
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1574272, 0.1647350, -0.1708696, 0.2229829, -0.3771502, 0.3312626
1: -0.0686271, 0.0689221, -0.0778513, 0.0773110, -0.1459381, 0.1467734
2: -0.1053974, 0.0966751, -0.1154248, 0.1155212, -0.2209186, 0.2121000
3: -0.0864605, 0.1418168, -0.0923715, 0.1550202, -0.2414807, 0.2341883
4: -0.0690144, 0.0566287, -0.0852941, 0.0655414, -0.1339202, 0.1414983
5: -0.0774258, 0.1084655, -0.0867782, 0.1221924, -0.1985891, 0.1924735
6: -0.0898071, 0.0869272, -0.0974704, 0.0960724, -0.1826857, 0.1810776
7: -0.0800737, 0.0672017, -0.0868933, 0.0782940, -0.1583677, 0.1540950
8: 0.5648291, 1.0987828, 0.5048499, 1.1085424, -0.5167670, 0.5753944
9: -0.0507631, 0.1523322, -0.0635459, 0.1620800, -0.1992641, 0.2015342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 59
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 142
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 255
type: B, layer: 3, pos: 140
type: B, layer: 3, pos: 177
type: B, layer: 3, pos: 248
type: B, layer: 3, pos: 121
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 91
type: B, layer: 3, pos: 180

Time for candidate selection: 4.22 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3584130
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3618979
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.1600223, 0.1761911, -0.1708696, 0.2229829, -0.3792681, 0.3424203
1: -0.0703633, 0.0695929, -0.0778513, 0.0773110, -0.1476743, 0.1474442
2: -0.1073237, 0.0999089, -0.1154248, 0.1155212, -0.2228449, 0.2153337
3: -0.0875995, 0.1429007, -0.0923715, 0.1550202, -0.2426196, 0.2352722
4: -0.0714792, 0.0582858, -0.0852941, 0.0655414, -0.1363571, 0.1430888
5: -0.0791898, 0.1096610, -0.0867782, 0.1221924, -0.2000332, 0.1938493
6: -0.0912108, 0.0878678, -0.0974704, 0.0960724, -0.1839260, 0.1819101
7: -0.0807419, 0.0691984, -0.0868933, 0.0782940, -0.1590359, 0.1560917
8: 0.5550525, 1.1008596, 0.5048499, 1.1085424, -0.5262451, 0.5747843
9: -0.0531513, 0.1534001, -0.0635459, 0.1620800, -0.2005690, 0.2019295

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 59
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 142
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 255
type: B, layer: 3, pos: 140
type: B, layer: 3, pos: 177
type: B, layer: 3, pos: 248
type: B, layer: 3, pos: 121
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 91
type: B, layer: 3, pos: 180

Time for candidate selection: 4.23 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3584130
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3618980
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1831207, 0.1761303, -0.1574272, 0.1647350, -0.3432420, 0.3290908
1: -0.0819185, 0.0729741, -0.0686271, 0.0689221, -0.1508406, 0.1416012
2: -0.1274372, 0.1099185, -0.1053974, 0.0966751, -0.2241123, 0.2153158
3: -0.1037588, 0.1409508, -0.0864605, 0.1418168, -0.2455755, 0.2274113
4: -0.0729626, 0.0699146, -0.0690144, 0.0566287, -0.1295006, 0.1382553
5: -0.0953397, 0.1077655, -0.0774258, 0.1084655, -0.2009420, 0.1851912
6: -0.1057247, 0.0868782, -0.0898071, 0.0869272, -0.1891862, 0.1766853
7: -0.0799017, 0.0852706, -0.0800737, 0.0672017, -0.1471034, 0.1653443
8: 0.5579582, 1.1290481, 0.5648291, 1.0987828, -0.5092549, 0.5335672
9: -0.0723124, 0.1527038, -0.0507631, 0.1523322, -0.2096415, 0.1889861

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3324947, upper bound: 0.2893766
time: 0.87 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 59
type: B, layer: 3, pos: 142
type: B, layer: 3, pos: 255
type: B, layer: 3, pos: 177
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 121
type: B, layer: 3, pos: 248
type: B, layer: 3, pos: 180

Time for candidate selection: 5.07 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3583346, upper bound: 0.3479309
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3571132, upper bound: 0.3451735
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.1874533, 0.1940520, -0.1574272, 0.1647350, -0.3476967, 0.3481534
1: -0.0847268, 0.0755529, -0.0686271, 0.0689221, -0.1536489, 0.1441800
2: -0.1307435, 0.1143026, -0.1053974, 0.0966751, -0.2274187, 0.2197000
3: -0.1058973, 0.1444872, -0.0864605, 0.1418168, -0.2477140, 0.2309477
4: -0.0784101, 0.0726651, -0.0690144, 0.0566287, -0.1350121, 0.1410232
5: -0.0983221, 0.1114792, -0.0774258, 0.1084655, -0.2041430, 0.1889050
6: -0.1082967, 0.0894233, -0.0898071, 0.0869272, -0.1918436, 0.1792304
7: -0.0817710, 0.0888494, -0.0800737, 0.0672017, -0.1489727, 0.1689232
8: 0.5407375, 1.1325693, 0.5648291, 1.0987828, -0.5310884, 0.5385699
9: -0.0764745, 0.1554998, -0.0507631, 0.1523322, -0.2141415, 0.1935951

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3324947, upper bound: 0.2893766
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 59
type: B, layer: 3, pos: 142
type: B, layer: 3, pos: 255
type: B, layer: 3, pos: 177
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 121
type: B, layer: 3, pos: 248
type: B, layer: 3, pos: 180

Time for candidate selection: 5.06 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3609453, upper bound: 0.3479309
time: 0.93 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3609453, upper bound: 0.3451735
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1831207, 0.1761303, -0.1600223, 0.1761911, -0.3550415, 0.3319945
1: -0.0819185, 0.0729741, -0.0703633, 0.0695929, -0.1515114, 0.1433374
2: -0.1274372, 0.1099185, -0.1073237, 0.0999089, -0.2273460, 0.2172422
3: -0.1037588, 0.1409508, -0.0875995, 0.1429007, -0.2466595, 0.2285503
4: -0.0729626, 0.0699146, -0.0714792, 0.0582858, -0.1311990, 0.1407785
5: -0.0953397, 0.1077655, -0.0791898, 0.1096610, -0.2028823, 0.1869553
6: -0.1057247, 0.0868782, -0.0912108, 0.0878678, -0.1903858, 0.1780890
7: -0.0799017, 0.0852706, -0.0807419, 0.0691984, -0.1491001, 0.1660125
8: 0.5579582, 1.1290481, 0.5550525, 1.1008596, -0.5141537, 0.5476725
9: -0.0723124, 0.1527038, -0.0531513, 0.1534001, -0.2118225, 0.1922479

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3258166, upper bound: 0.2831905
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 59
type: B, layer: 3, pos: 142
type: B, layer: 3, pos: 255
type: B, layer: 3, pos: 177
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 121
type: B, layer: 3, pos: 248
type: B, layer: 3, pos: 180

Time for candidate selection: 5.05 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3571132, upper bound: 0.3451735
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3571132, upper bound: 0.3451735
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.1874533, 0.1940520, -0.1600223, 0.1761911, -0.3589484, 0.3503792
1: -0.0847268, 0.0755529, -0.0703633, 0.0695929, -0.1543197, 0.1459162
2: -0.1307435, 0.1143026, -0.1073237, 0.0999089, -0.2306524, 0.2216264
3: -0.1058973, 0.1444872, -0.0875995, 0.1429007, -0.2487980, 0.2320866
4: -0.0784101, 0.0726651, -0.0714792, 0.0582858, -0.1365997, 0.1434723
5: -0.0983221, 0.1114792, -0.0791898, 0.1096610, -0.2056108, 0.1906690
6: -0.1082967, 0.0894233, -0.0912108, 0.0878678, -0.1927346, 0.1806341
7: -0.0817710, 0.0888494, -0.0807419, 0.0691984, -0.1509695, 0.1695913
8: 0.5407375, 1.1325693, 0.5550525, 1.1008596, -0.5320849, 0.5488865
9: -0.0764745, 0.1554998, -0.0531513, 0.1534001, -0.2148327, 0.1946121

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3258166, upper bound: 0.2831905
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 59
type: B, layer: 3, pos: 142
type: B, layer: 3, pos: 255
type: B, layer: 3, pos: 177
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 121
type: B, layer: 3, pos: 248
type: B, layer: 3, pos: 180

Time for candidate selection: 5.07 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3609453, upper bound: 0.3451735
time: 0.93 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3609453, upper bound: 0.3451735
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1825894, 0.2108650, -0.1775626, 0.1601652, -0.3376615, 0.3852074
1: -0.0848782, 0.0761132, -0.0783530, 0.0695895, -0.1544677, 0.1544662
2: -0.1258671, 0.1177361, -0.1232050, 0.1047693, -0.2306365, 0.2409411
3: -0.1002077, 0.1429518, -0.1009970, 0.1418455, -0.2420532, 0.2439488
4: -0.0836409, 0.0729752, -0.0690137, 0.0663872, -0.1500281, 0.1414520
5: -0.0952044, 0.1099963, -0.0915928, 0.1085127, -0.2037170, 0.2015891
6: -0.1059879, 0.0886960, -0.1023265, 0.0869845, -0.1929439, 0.1910225
7: -0.0812051, 0.0887380, -0.0800880, 0.0805176, -0.1617226, 0.1688260
8: 0.5332632, 1.1203173, 0.5668604, 1.1246995, -0.5687873, 0.5192652
9: -0.0758681, 0.1547121, -0.0669128, 0.1525115, -0.2122564, 0.2081864

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3154721, upper bound: 0.2386784
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 59
type: B, layer: 3, pos: 255
type: B, layer: 3, pos: 142
type: B, layer: 3, pos: 177
type: B, layer: 3, pos: 248
type: B, layer: 3, pos: 42

Time for candidate selection: 3.04 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3550109, upper bound: 0.3219096
time: 1.14 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3550109, upper bound: 0.3219343
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1864149, 0.2314227, -0.1775626, 0.1601652, -0.3418088, 0.4058890
1: -0.0874691, 0.0785330, -0.0783530, 0.0695895, -0.1570586, 0.1568860
2: -0.1287732, 0.1228720, -0.1232050, 0.1047693, -0.2335425, 0.2460769
3: -0.1019717, 0.1476737, -0.1009970, 0.1418455, -0.2438172, 0.2486707
4: -0.0896836, 0.0754122, -0.0690137, 0.0663872, -0.1560435, 0.1439333
5: -0.0979362, 0.1148993, -0.0915928, 0.1085127, -0.2064488, 0.2064921
6: -0.1081337, 0.0919547, -0.1023265, 0.0869845, -0.1951182, 0.1942812
7: -0.0836563, 0.0919675, -0.0800880, 0.0805176, -0.1641739, 0.1720555
8: 0.5115326, 1.1234193, 0.5668604, 1.1246995, -0.5923040, 0.5247712
9: -0.0795045, 0.1581024, -0.0669128, 0.1525115, -0.2164650, 0.2111062

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3023270, upper bound: 0.1997539
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 59
type: B, layer: 3, pos: 255
type: B, layer: 3, pos: 142
type: B, layer: 3, pos: 177
type: B, layer: 3, pos: 248
type: B, layer: 3, pos: 42

Time for candidate selection: 3.01 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3575774, upper bound: 0.3219096
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3575774, upper bound: 0.3219096
time: 0.98 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1660702, 0.2018342, -0.1574272, 0.1647350, -0.3262532, 0.3551338
1: -0.0747557, 0.0741434, -0.0686271, 0.0689221, -0.1436778, 0.1427705
2: -0.1117253, 0.1088224, -0.1053974, 0.0966751, -0.2084005, 0.2142198
3: -0.0900854, 0.1500339, -0.0864605, 0.1418168, -0.2319021, 0.2364944
4: -0.0792971, 0.0625792, -0.0690144, 0.0566287, -0.1353558, 0.1309292
5: -0.0834441, 0.1170112, -0.0774258, 0.1084655, -0.1888929, 0.1925342
6: -0.0948899, 0.0926307, -0.0898071, 0.0869272, -0.1783734, 0.1790389
7: -0.0843168, 0.0746323, -0.0800737, 0.0672017, -0.1515185, 0.1547060
8: 0.5274178, 1.1043098, 0.5648291, 1.0987828, -0.5471997, 0.5106981
9: -0.0594005, 0.1584760, -0.0507631, 0.1523322, -0.1967569, 0.1946074

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2962512, upper bound: 0.2141893
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 59
type: B, layer: 3, pos: 142
type: B, layer: 3, pos: 255
type: B, layer: 3, pos: 177
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 121
type: B, layer: 3, pos: 248
type: B, layer: 3, pos: 180

Time for candidate selection: 5.02 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3595635, upper bound: 0.3445386
time: 0.93 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3584130, upper bound: 0.3427173
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1708696, 0.2229829, -0.1574272, 0.1647350, -0.3312626, 0.3771502
1: -0.0778513, 0.0773110, -0.0686271, 0.0689221, -0.1467734, 0.1459381
2: -0.1154248, 0.1155212, -0.1053974, 0.0966751, -0.2121000, 0.2209186
3: -0.0923715, 0.1550202, -0.0864605, 0.1418168, -0.2341883, 0.2414807
4: -0.0852941, 0.0655414, -0.0690144, 0.0566287, -0.1414983, 0.1339202
5: -0.0867782, 0.1221924, -0.0774258, 0.1084655, -0.1924734, 0.1985891
6: -0.0974704, 0.0960724, -0.0898071, 0.0869272, -0.1810776, 0.1826857
7: -0.0868933, 0.0782940, -0.0800737, 0.0672017, -0.1540950, 0.1583677
8: 0.5048499, 1.1085424, 0.5648291, 1.0987828, -0.5753942, 0.5167670
9: -0.0635459, 0.1620800, -0.0507631, 0.1523322, -0.2015341, 0.1992642

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2962512, upper bound: 0.2141893
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 59
type: B, layer: 3, pos: 142
type: B, layer: 3, pos: 255
type: B, layer: 3, pos: 177
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 121
type: B, layer: 3, pos: 248
type: B, layer: 3, pos: 180

Time for candidate selection: 4.98 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3595635, upper bound: 0.3445386
time: 0.93 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3584130, upper bound: 0.3427173
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1660702, 0.2018342, -0.1600223, 0.1761911, -0.3380329, 0.3579617
1: -0.0747557, 0.0741434, -0.0703633, 0.0695929, -0.1443486, 0.1445067
2: -0.1117253, 0.1088224, -0.1073237, 0.0999089, -0.2116342, 0.2161461
3: -0.0900854, 0.1500339, -0.0875995, 0.1429007, -0.2329861, 0.2376333
4: -0.0792971, 0.0625792, -0.0714792, 0.0582858, -0.1370451, 0.1334496
5: -0.0834441, 0.1170112, -0.0791898, 0.1096610, -0.1907832, 0.1944954
6: -0.0948899, 0.0926307, -0.0912108, 0.0878678, -0.1795070, 0.1805897
7: -0.0843168, 0.0746323, -0.0807419, 0.0691984, -0.1535152, 0.1553742
8: 0.5274178, 1.1043098, 0.5550525, 1.1008596, -0.5512948, 0.5242290
9: -0.0594005, 0.1584760, -0.0531513, 0.1534001, -0.1986885, 0.1975968

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 59
type: B, layer: 3, pos: 142
type: B, layer: 3, pos: 255
type: B, layer: 3, pos: 177
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 121
type: B, layer: 3, pos: 248
type: B, layer: 3, pos: 180

Time for candidate selection: 4.30 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3584130, upper bound: 0.3427173
time: 0.91 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3584130, upper bound: 0.3427173
time: 0.96 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1708696, 0.2229829, -0.1600223, 0.1761911, -0.3424202, 0.3792679
1: -0.0778513, 0.0773110, -0.0703633, 0.0695929, -0.1474442, 0.1476743
2: -0.1154248, 0.1155212, -0.1073237, 0.0999089, -0.2153337, 0.2228449
3: -0.0923715, 0.1550202, -0.0875995, 0.1429007, -0.2352722, 0.2426196
4: -0.0852941, 0.0655414, -0.0714792, 0.0582858, -0.1430888, 0.1363571
5: -0.0867782, 0.1221924, -0.0791898, 0.1096610, -0.1938493, 0.2000332
6: -0.0974704, 0.0960724, -0.0912108, 0.0878678, -0.1819101, 0.1839260
7: -0.0868933, 0.0782940, -0.0807419, 0.0691984, -0.1560917, 0.1590359
8: 0.5048499, 1.1085424, 0.5550525, 1.1008596, -0.5747843, 0.5262451
9: -0.0635459, 0.1620800, -0.0531513, 0.1534001, -0.2019295, 0.2005691

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 59
type: B, layer: 3, pos: 142
type: B, layer: 3, pos: 255
type: B, layer: 3, pos: 177
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 121
type: B, layer: 3, pos: 248
type: B, layer: 3, pos: 180

Time for candidate selection: 4.35 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3584130, upper bound: 0.3427173
time: 0.92 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3584130, upper bound: 0.3427173
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1660702, 0.2018342, -0.1574272, 0.1647350, -0.3262532, 0.3551338
1: -0.0747557, 0.0741434, -0.0686271, 0.0689221, -0.1436778, 0.1427705
2: -0.1117253, 0.1088224, -0.1053974, 0.0966751, -0.2084005, 0.2142198
3: -0.0900854, 0.1500339, -0.0864605, 0.1418168, -0.2319021, 0.2364944
4: -0.0792971, 0.0625792, -0.0690144, 0.0566287, -0.1353558, 0.1309292
5: -0.0834441, 0.1170112, -0.0774258, 0.1084655, -0.1888929, 0.1925342
6: -0.0948899, 0.0926307, -0.0898071, 0.0869272, -0.1783734, 0.1790389
7: -0.0843168, 0.0746323, -0.0800737, 0.0672017, -0.1515185, 0.1547060
8: 0.5274178, 1.1043098, 0.5648291, 1.0987828, -0.5471997, 0.5106981
9: -0.0594005, 0.1584760, -0.0507631, 0.1523322, -0.1967569, 0.1946074

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2962512, upper bound: 0.2141893
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 59
type: B, layer: 3, pos: 142
type: B, layer: 3, pos: 255
type: B, layer: 3, pos: 177
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 121
type: B, layer: 3, pos: 248
type: B, layer: 3, pos: 180

Time for candidate selection: 4.95 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3595635, upper bound: 0.3445386
time: 0.98 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3584130, upper bound: 0.3427173
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.1708696, 0.2229829, -0.1574272, 0.1647350, -0.3312626, 0.3771502
1: -0.0778513, 0.0773110, -0.0686271, 0.0689221, -0.1467734, 0.1459381
2: -0.1154248, 0.1155212, -0.1053974, 0.0966751, -0.2121000, 0.2209186
3: -0.0923715, 0.1550202, -0.0864605, 0.1418168, -0.2341883, 0.2414807
4: -0.0852941, 0.0655414, -0.0690144, 0.0566287, -0.1414983, 0.1339202
5: -0.0867782, 0.1221924, -0.0774258, 0.1084655, -0.1924734, 0.1985891
6: -0.0974704, 0.0960724, -0.0898071, 0.0869272, -0.1810776, 0.1826857
7: -0.0868933, 0.0782940, -0.0800737, 0.0672017, -0.1540950, 0.1583677
8: 0.5048499, 1.1085424, 0.5648291, 1.0987828, -0.5753942, 0.5167670
9: -0.0635459, 0.1620800, -0.0507631, 0.1523322, -0.2015341, 0.1992642

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2962512, upper bound: 0.2141893
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 59
type: B, layer: 3, pos: 142
type: B, layer: 3, pos: 255
type: B, layer: 3, pos: 177
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 121
type: B, layer: 3, pos: 248
type: B, layer: 3, pos: 180

Time for candidate selection: 5.05 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3618980, upper bound: 0.3445386
time: 0.93 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3618980, upper bound: 0.3427173
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1660702, 0.2018342, -0.1600223, 0.1761911, -0.3380329, 0.3579617
1: -0.0747557, 0.0741434, -0.0703633, 0.0695929, -0.1443486, 0.1445067
2: -0.1117253, 0.1088224, -0.1073237, 0.0999089, -0.2116342, 0.2161461
3: -0.0900854, 0.1500339, -0.0875995, 0.1429007, -0.2329861, 0.2376333
4: -0.0792971, 0.0625792, -0.0714792, 0.0582858, -0.1370451, 0.1334496
5: -0.0834441, 0.1170112, -0.0791898, 0.1096610, -0.1907832, 0.1944954
6: -0.0948899, 0.0926307, -0.0912108, 0.0878678, -0.1795070, 0.1805897
7: -0.0843168, 0.0746323, -0.0807419, 0.0691984, -0.1535152, 0.1553742
8: 0.5274178, 1.1043098, 0.5550525, 1.1008596, -0.5512948, 0.5242290
9: -0.0594005, 0.1584760, -0.0531513, 0.1534001, -0.1986885, 0.1975968

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 59
type: B, layer: 3, pos: 142
type: B, layer: 3, pos: 255
type: B, layer: 3, pos: 177
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 121
type: B, layer: 3, pos: 248
type: B, layer: 3, pos: 180

Time for candidate selection: 4.31 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3584130, upper bound: 0.3427173
time: 0.92 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3584130, upper bound: 0.3427173
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.1708696, 0.2229829, -0.1600223, 0.1761911, -0.3424202, 0.3792679
1: -0.0778513, 0.0773110, -0.0703633, 0.0695929, -0.1474442, 0.1476743
2: -0.1154248, 0.1155212, -0.1073237, 0.0999089, -0.2153337, 0.2228449
3: -0.0923715, 0.1550202, -0.0875995, 0.1429007, -0.2352722, 0.2426196
4: -0.0852941, 0.0655414, -0.0714792, 0.0582858, -0.1430888, 0.1363571
5: -0.0867782, 0.1221924, -0.0791898, 0.1096610, -0.1938493, 0.2000332
6: -0.0974704, 0.0960724, -0.0912108, 0.0878678, -0.1819101, 0.1839260
7: -0.0868933, 0.0782940, -0.0807419, 0.0691984, -0.1560917, 0.1590359
8: 0.5048499, 1.1085424, 0.5550525, 1.1008596, -0.5747843, 0.5262451
9: -0.0635459, 0.1620800, -0.0531513, 0.1534001, -0.2019295, 0.2005691

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 144

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 154
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 116
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 59
type: B, layer: 3, pos: 142
type: B, layer: 3, pos: 255
type: B, layer: 3, pos: 177
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 121
type: B, layer: 3, pos: 248
type: B, layer: 3, pos: 180

Time for candidate selection: 4.34 seconds

### Candidate
type: B, layer: 3, pos: 29

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3618980, upper bound: 0.3427173
time: 0.93 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3618980, upper bound: 0.3427173
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1831207, 0.1761303, -0.1882254, 0.2024014, -0.3815420, 0.3597860
1: -0.0819185, 0.0729741, -0.0852387, 0.0759273, -0.1578458, 0.1582129
2: -0.1274372, 0.1099185, -0.1311661, 0.1160526, -0.2434898, 0.2410845
3: -0.1037588, 0.1409508, -0.1060354, 0.1522663, -0.2560251, 0.2469863
4: -0.0729626, 0.0699146, -0.0817750, 0.0731525, -0.1460103, 0.1513889
5: -0.0953397, 0.1077655, -0.0988369, 0.1193606, -0.2136198, 0.2066024
6: -0.1057247, 0.0868782, -0.1085465, 0.0942346, -0.1995564, 0.1954248
7: -0.0799017, 0.0852706, -0.0854496, 0.0892740, -0.1691756, 0.1707202
8: 0.5579582, 1.1290481, 0.5222746, 1.1329589, -0.5432761, 0.5774212
9: -0.0723124, 0.1527038, -0.0769448, 0.1604559, -0.2174237, 0.2147627

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3610570, upper bound: 0.3366729
time: 0.91 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3430140, upper bound: 0.3355150
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.1874533, 0.1940520, -0.1885679, 0.2049418, -0.3886856, 0.3784502
1: -0.0847268, 0.0755529, -0.0855051, 0.0763045, -0.1610313, 0.1610581
2: -0.1307435, 0.1143026, -0.1313922, 0.1166717, -0.2474152, 0.2456948
3: -0.1058973, 0.1444872, -0.1061361, 0.1530833, -0.2589806, 0.2506233
4: -0.0784101, 0.0726651, -0.0825749, 0.0734045, -0.1516517, 0.1549508
5: -0.0983221, 0.1114792, -0.0990732, 0.1202016, -0.2176023, 0.2105524
6: -0.1082967, 0.0894233, -0.1087306, 0.0947756, -0.2025538, 0.1981540
7: -0.0817710, 0.0888494, -0.0858559, 0.0895850, -0.1713560, 0.1747053
8: 0.5407375, 1.1325693, 0.5193495, 1.1331092, -0.5610108, 0.5857654
9: -0.0764745, 0.1554998, -0.0772883, 0.1610278, -0.2226081, 0.2172552

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3636719, upper bound: 0.3366729
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3453056, upper bound: 0.3355150
time: 0.72 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 2.78 seconds
IS_A1_B1_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3432554, upper bound: 0.3378745
IS_A1_B1_A1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3212425, upper bound: 0.3362557
IS_A1_B1_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3383572, upper bound: 0.3352467
IS_A1_B1_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3178940, upper bound: 0.3325054
IS_A1_B2_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3584130
IS_A1_B2_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3584130
IS_A1_B2_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3584130
IS_A1_B2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3584130
IS_A1_B2_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3584130
IS_A1_B2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3618979
IS_A1_B2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3584130
IS_A1_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3618979
IS_A1_B2_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3584130
IS_A1_B2_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3584130
IS_A1_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3584130
IS_A1_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3584130
IS_A1_B2_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3584130
IS_A1_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3618979
IS_A1_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3584130
IS_A1_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3618980
IS_A2_B1_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3583346, upper bound: 0.3479309
IS_A2_B1_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3571132, upper bound: 0.3451735
IS_A2_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3609453, upper bound: 0.3479309
IS_A2_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3609453, upper bound: 0.3451735
IS_A2_B1_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3571132, upper bound: 0.3451735
IS_A2_B1_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3571132, upper bound: 0.3451735
IS_A2_B1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3609453, upper bound: 0.3451735
IS_A2_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3609453, upper bound: 0.3451735
IS_A2_B1_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3550109, upper bound: 0.3219096
IS_A2_B1_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3550109, upper bound: 0.3219343
IS_A2_B1_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3575774, upper bound: 0.3219096
IS_A2_B1_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3575774, upper bound: 0.3219096
IS_A2_B1_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3595635, upper bound: 0.3445386
IS_A2_B1_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3584130, upper bound: 0.3427173
IS_A2_B1_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3595635, upper bound: 0.3445386
IS_A2_B1_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3584130, upper bound: 0.3427173
IS_A2_B1_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3584130, upper bound: 0.3427173
IS_A2_B1_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3584130, upper bound: 0.3427173
IS_A2_B1_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3584130, upper bound: 0.3427173
IS_A2_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3584130, upper bound: 0.3427173
IS_A2_B1_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3595635, upper bound: 0.3445386
IS_A2_B1_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3584130, upper bound: 0.3427173
IS_A2_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3618980, upper bound: 0.3445386
IS_A2_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3618980, upper bound: 0.3427173
IS_A2_B1_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3584130, upper bound: 0.3427173
IS_A2_B1_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3584130, upper bound: 0.3427173
IS_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3618980, upper bound: 0.3427173
IS_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3618980, upper bound: 0.3427173
IS_A2_B2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3610570, upper bound: 0.3366729
IS_A2_B2_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3430140, upper bound: 0.3355150
IS_A2_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3636719, upper bound: 0.3366729
IS_A2_B2_A1_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.78
Output dim: 8, lower bound: -0.3453056, upper bound: 0.3355150

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1574272, 0.1647350, -0.1660702, 0.2018342, -0.3551338, 0.3262532
1: -0.0686271, 0.0689221, -0.0747557, 0.0741434, -0.1427705, 0.1436778
2: -0.1053974, 0.0966751, -0.1117253, 0.1088224, -0.2142198, 0.2084005
3: -0.0864605, 0.1418168, -0.0900854, 0.1500339, -0.2364944, 0.2319021
4: -0.0690144, 0.0566287, -0.0792971, 0.0625792, -0.1309292, 0.1353558
5: -0.0774258, 0.1084655, -0.0834441, 0.1170112, -0.1925342, 0.1888929
6: -0.0898071, 0.0869272, -0.0948899, 0.0926307, -0.1790389, 0.1783733
7: -0.0800737, 0.0672017, -0.0843168, 0.0746323, -0.1547060, 0.1515185
8: 0.5648291, 1.0987828, 0.5274178, 1.1043098, -0.5106983, 0.5472000
9: -0.0507631, 0.1523322, -0.0594005, 0.1584760, -0.1946074, 0.1967568

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2141893, upper bound: 0.2962512
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 59
type: A, layer: 3, pos: 142
type: A, layer: 3, pos: 255
type: A, layer: 3, pos: 177
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 121
type: A, layer: 3, pos: 248
type: A, layer: 3, pos: 180

Time for candidate selection: 4.91 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3445386, upper bound: 0.3595635
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3584130
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1574272, 0.1647350, -0.1708696, 0.2229829, -0.3771502, 0.3312626
1: -0.0686271, 0.0689221, -0.0778513, 0.0773110, -0.1459381, 0.1467734
2: -0.1053974, 0.0966751, -0.1154248, 0.1155212, -0.2209186, 0.2121000
3: -0.0864605, 0.1418168, -0.0923715, 0.1550202, -0.2414807, 0.2341883
4: -0.0690144, 0.0566287, -0.0852941, 0.0655414, -0.1339202, 0.1414983
5: -0.0774258, 0.1084655, -0.0867782, 0.1221924, -0.1985891, 0.1924735
6: -0.0898071, 0.0869272, -0.0974704, 0.0960724, -0.1826857, 0.1810776
7: -0.0800737, 0.0672017, -0.0868933, 0.0782940, -0.1583677, 0.1540950
8: 0.5648291, 1.0987828, 0.5048499, 1.1085424, -0.5167670, 0.5753944
9: -0.0507631, 0.1523322, -0.0635459, 0.1620800, -0.1992641, 0.2015342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2141893, upper bound: 0.2962512
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 59
type: A, layer: 3, pos: 142
type: A, layer: 3, pos: 255
type: A, layer: 3, pos: 177
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 121
type: A, layer: 3, pos: 248
type: A, layer: 3, pos: 180

Time for candidate selection: 4.91 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3445386, upper bound: 0.3595635
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3584130
time: 1.07 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.1600223, 0.1761911, -0.1660702, 0.2018342, -0.3579617, 0.3380330
1: -0.0703633, 0.0695929, -0.0747557, 0.0741434, -0.1445067, 0.1443486
2: -0.1073237, 0.0999089, -0.1117253, 0.1088224, -0.2161461, 0.2116342
3: -0.0875995, 0.1429007, -0.0900854, 0.1500339, -0.2376333, 0.2329861
4: -0.0714792, 0.0582858, -0.0792971, 0.0625792, -0.1334496, 0.1370451
5: -0.0791898, 0.1096610, -0.0834441, 0.1170112, -0.1944954, 0.1907833
6: -0.0912108, 0.0878678, -0.0948899, 0.0926307, -0.1805897, 0.1795070
7: -0.0807419, 0.0691984, -0.0843168, 0.0746323, -0.1553742, 0.1535152
8: 0.5550525, 1.1008596, 0.5274178, 1.1043098, -0.5242290, 0.5512948
9: -0.0531513, 0.1534001, -0.0594005, 0.1584760, -0.1975967, 0.1986885

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 59
type: A, layer: 3, pos: 142
type: A, layer: 3, pos: 255
type: A, layer: 3, pos: 177
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 121
type: A, layer: 3, pos: 248
type: A, layer: 3, pos: 180

Time for candidate selection: 4.39 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3584130
time: 1.08 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3584130
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.1600223, 0.1761911, -0.1708696, 0.2229829, -0.3792681, 0.3424203
1: -0.0703633, 0.0695929, -0.0778513, 0.0773110, -0.1476743, 0.1474442
2: -0.1073237, 0.0999089, -0.1154248, 0.1155212, -0.2228449, 0.2153337
3: -0.0875995, 0.1429007, -0.0923715, 0.1550202, -0.2426196, 0.2352722
4: -0.0714792, 0.0582858, -0.0852941, 0.0655414, -0.1363571, 0.1430888
5: -0.0791898, 0.1096610, -0.0867782, 0.1221924, -0.2000332, 0.1938493
6: -0.0912108, 0.0878678, -0.0974704, 0.0960724, -0.1839260, 0.1819101
7: -0.0807419, 0.0691984, -0.0868933, 0.0782940, -0.1590359, 0.1560917
8: 0.5550525, 1.1008596, 0.5048499, 1.1085424, -0.5262451, 0.5747843
9: -0.0531513, 0.1534001, -0.0635459, 0.1620800, -0.2005690, 0.2019295

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 59
type: A, layer: 3, pos: 142
type: A, layer: 3, pos: 255
type: A, layer: 3, pos: 177
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 121
type: A, layer: 3, pos: 248
type: A, layer: 3, pos: 180

Time for candidate selection: 4.34 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3584130
time: 1.07 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3584130
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1574272, 0.1647350, -0.1660702, 0.2018342, -0.3551338, 0.3262532
1: -0.0686271, 0.0689221, -0.0747557, 0.0741434, -0.1427705, 0.1436778
2: -0.1053974, 0.0966751, -0.1117253, 0.1088224, -0.2142198, 0.2084005
3: -0.0864605, 0.1418168, -0.0900854, 0.1500339, -0.2364944, 0.2319021
4: -0.0690144, 0.0566287, -0.0792971, 0.0625792, -0.1309292, 0.1353558
5: -0.0774258, 0.1084655, -0.0834441, 0.1170112, -0.1925342, 0.1888929
6: -0.0898071, 0.0869272, -0.0948899, 0.0926307, -0.1790389, 0.1783733
7: -0.0800737, 0.0672017, -0.0843168, 0.0746323, -0.1547060, 0.1515185
8: 0.5648291, 1.0987828, 0.5274178, 1.1043098, -0.5106983, 0.5472000
9: -0.0507631, 0.1523322, -0.0594005, 0.1584760, -0.1946074, 0.1967568

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2141893, upper bound: 0.2962512
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 59
type: A, layer: 3, pos: 142
type: A, layer: 3, pos: 255
type: A, layer: 3, pos: 177
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 121
type: A, layer: 3, pos: 248
type: A, layer: 3, pos: 180

Time for candidate selection: 5.03 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3445386, upper bound: 0.3595635
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3584130
time: 1.16 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1574272, 0.1647350, -0.1708696, 0.2229829, -0.3771502, 0.3312626
1: -0.0686271, 0.0689221, -0.0778513, 0.0773110, -0.1459381, 0.1467734
2: -0.1053974, 0.0966751, -0.1154248, 0.1155212, -0.2209186, 0.2121000
3: -0.0864605, 0.1418168, -0.0923715, 0.1550202, -0.2414807, 0.2341883
4: -0.0690144, 0.0566287, -0.0852941, 0.0655414, -0.1339202, 0.1414983
5: -0.0774258, 0.1084655, -0.0867782, 0.1221924, -0.1985891, 0.1924735
6: -0.0898071, 0.0869272, -0.0974704, 0.0960724, -0.1826857, 0.1810776
7: -0.0800737, 0.0672017, -0.0868933, 0.0782940, -0.1583677, 0.1540950
8: 0.5648291, 1.0987828, 0.5048499, 1.1085424, -0.5167670, 0.5753944
9: -0.0507631, 0.1523322, -0.0635459, 0.1620800, -0.1992641, 0.2015342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2141893, upper bound: 0.2962512
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 59
type: A, layer: 3, pos: 142
type: A, layer: 3, pos: 255
type: A, layer: 3, pos: 177
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 121
type: A, layer: 3, pos: 248
type: A, layer: 3, pos: 180

Time for candidate selection: 5.02 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3445386, upper bound: 0.3618980
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3618980
time: 1.82 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.1600223, 0.1761911, -0.1660702, 0.2018342, -0.3579617, 0.3380330
1: -0.0703633, 0.0695929, -0.0747557, 0.0741434, -0.1445067, 0.1443486
2: -0.1073237, 0.0999089, -0.1117253, 0.1088224, -0.2161461, 0.2116342
3: -0.0875995, 0.1429007, -0.0900854, 0.1500339, -0.2376333, 0.2329861
4: -0.0714792, 0.0582858, -0.0792971, 0.0625792, -0.1334496, 0.1370451
5: -0.0791898, 0.1096610, -0.0834441, 0.1170112, -0.1944954, 0.1907833
6: -0.0912108, 0.0878678, -0.0948899, 0.0926307, -0.1805897, 0.1795070
7: -0.0807419, 0.0691984, -0.0843168, 0.0746323, -0.1553742, 0.1535152
8: 0.5550525, 1.1008596, 0.5274178, 1.1043098, -0.5242290, 0.5512948
9: -0.0531513, 0.1534001, -0.0594005, 0.1584760, -0.1975967, 0.1986885

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 59
type: A, layer: 3, pos: 142
type: A, layer: 3, pos: 255
type: A, layer: 3, pos: 177
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 121
type: A, layer: 3, pos: 248
type: A, layer: 3, pos: 180

Time for candidate selection: 4.29 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3584130
time: 1.07 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3584130
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.1600223, 0.1761911, -0.1708696, 0.2229829, -0.3792681, 0.3424203
1: -0.0703633, 0.0695929, -0.0778513, 0.0773110, -0.1476743, 0.1474442
2: -0.1073237, 0.0999089, -0.1154248, 0.1155212, -0.2228449, 0.2153337
3: -0.0875995, 0.1429007, -0.0923715, 0.1550202, -0.2426196, 0.2352722
4: -0.0714792, 0.0582858, -0.0852941, 0.0655414, -0.1363571, 0.1430888
5: -0.0791898, 0.1096610, -0.0867782, 0.1221924, -0.2000332, 0.1938493
6: -0.0912108, 0.0878678, -0.0974704, 0.0960724, -0.1839260, 0.1819101
7: -0.0807419, 0.0691984, -0.0868933, 0.0782940, -0.1590359, 0.1560917
8: 0.5550525, 1.1008596, 0.5048499, 1.1085424, -0.5262451, 0.5747843
9: -0.0531513, 0.1534001, -0.0635459, 0.1620800, -0.2005690, 0.2019295

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 59
type: A, layer: 3, pos: 142
type: A, layer: 3, pos: 255
type: A, layer: 3, pos: 177
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 121
type: A, layer: 3, pos: 248
type: A, layer: 3, pos: 180

Time for candidate selection: 4.34 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3618980
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3618980
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1574272, 0.1647350, -0.1660702, 0.2018342, -0.3551338, 0.3262532
1: -0.0686271, 0.0689221, -0.0747557, 0.0741434, -0.1427705, 0.1436778
2: -0.1053974, 0.0966751, -0.1117253, 0.1088224, -0.2142198, 0.2084005
3: -0.0864605, 0.1418168, -0.0900854, 0.1500339, -0.2364944, 0.2319021
4: -0.0690144, 0.0566287, -0.0792971, 0.0625792, -0.1309292, 0.1353558
5: -0.0774258, 0.1084655, -0.0834441, 0.1170112, -0.1925342, 0.1888929
6: -0.0898071, 0.0869272, -0.0948899, 0.0926307, -0.1790389, 0.1783733
7: -0.0800737, 0.0672017, -0.0843168, 0.0746323, -0.1547060, 0.1515185
8: 0.5648291, 1.0987828, 0.5274178, 1.1043098, -0.5106983, 0.5472000
9: -0.0507631, 0.1523322, -0.0594005, 0.1584760, -0.1946074, 0.1967568

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2141893, upper bound: 0.2962512
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 59
type: A, layer: 3, pos: 142
type: A, layer: 3, pos: 255
type: A, layer: 3, pos: 177
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 121
type: A, layer: 3, pos: 248
type: A, layer: 3, pos: 180

Time for candidate selection: 4.86 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3445386, upper bound: 0.3595635
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3584130
time: 1.09 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1574272, 0.1647350, -0.1708696, 0.2229829, -0.3771502, 0.3312626
1: -0.0686271, 0.0689221, -0.0778513, 0.0773110, -0.1459381, 0.1467734
2: -0.1053974, 0.0966751, -0.1154248, 0.1155212, -0.2209186, 0.2121000
3: -0.0864605, 0.1418168, -0.0923715, 0.1550202, -0.2414807, 0.2341883
4: -0.0690144, 0.0566287, -0.0852941, 0.0655414, -0.1339202, 0.1414983
5: -0.0774258, 0.1084655, -0.0867782, 0.1221924, -0.1985891, 0.1924735
6: -0.0898071, 0.0869272, -0.0974704, 0.0960724, -0.1826857, 0.1810776
7: -0.0800737, 0.0672017, -0.0868933, 0.0782940, -0.1583677, 0.1540950
8: 0.5648291, 1.0987828, 0.5048499, 1.1085424, -0.5167670, 0.5753944
9: -0.0507631, 0.1523322, -0.0635459, 0.1620800, -0.1992641, 0.2015342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.2141893, upper bound: 0.2962512
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 59
type: A, layer: 3, pos: 142
type: A, layer: 3, pos: 255
type: A, layer: 3, pos: 177
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 121
type: A, layer: 3, pos: 248
type: A, layer: 3, pos: 180

Time for candidate selection: 4.93 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3445386, upper bound: 0.3595635
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3584130
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.1600223, 0.1761911, -0.1660702, 0.2018342, -0.3579617, 0.3380330
1: -0.0703633, 0.0695929, -0.0747557, 0.0741434, -0.1445067, 0.1443486
2: -0.1073237, 0.0999089, -0.1117253, 0.1088224, -0.2161461, 0.2116342
3: -0.0875995, 0.1429007, -0.0900854, 0.1500339, -0.2376333, 0.2329861
4: -0.0714792, 0.0582858, -0.0792971, 0.0625792, -0.1334496, 0.1370451
5: -0.0791898, 0.1096610, -0.0834441, 0.1170112, -0.1944954, 0.1907833
6: -0.0912108, 0.0878678, -0.0948899, 0.0926307, -0.1805897, 0.1795070
7: -0.0807419, 0.0691984, -0.0843168, 0.0746323, -0.1553742, 0.1535152
8: 0.5550525, 1.1008596, 0.5274178, 1.1043098, -0.5242290, 0.5512948
9: -0.0531513, 0.1534001, -0.0594005, 0.1584760, -0.1975967, 0.1986885

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 144

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 154
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 116
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 59
type: A, layer: 3, pos: 142
type: A, layer: 3, pos: 255
type: A, layer: 3, pos: 177
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 121
type: A, layer: 3, pos: 248
type: A, layer: 3, pos: 180

Time for candidate selection: 4.32 seconds

### Candidate
type: A, layer: 3, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3584130
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3427173, upper bound: 0.3584130
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.1600223, 0.1761911, -0.1708696, 0.2229829, -0.3792681, 0.3424203
1: -0.0703633, 0.0695929, -0.0778513, 0.0773110, -0.1476743, 0.1474442
2: -0.1073237, 0.0999089, -0.1154248, 0.1155212, -0.2228449, 0.2153337
3: -0.0875995, 0.1429007, -0.0923715, 0.1550202, -0.2426196, 0.2352722
4: -0.0714792, 0.0582858, -0.0852941, 0.0655414, -0.1363571, 0.1430888
5: -0.0791898, 0.1096610, -0.0867782, 0.1221924, -0.2000332, 0.1938493
6: -0.0912108, 0.0878678, -0.0974704, 0.0960724, -0.1839260, 0.1819101
7: -0.0807419, 0.0691984, -0.0868933, 0.0782940, -0.1590359, 0.1560917
8: 0.5550525, 1.1008596, 0.5048499, 1.1085424, -0.5262451, 0.5747843
9: -0.0531513, 0.1534001, -0.0635459, 0.1620800, -0.2005690, 0.2019295

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.16 seconds

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 2.91 + 597.98 = 600.88 seconds
